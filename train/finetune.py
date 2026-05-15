"""
Production GLiNER fine-tuning for PII detection.
All entity types are sourced exclusively from entities_config.yaml.

Two modes selected automatically based on available hardware:
  GPU  (CUDA / MPS) — mixed-precision, larger batch, more epochs, faster
  CPU               — smaller batch, fewer epochs, gradient accumulation

Run after generate_data.py:
    python train/finetune.py [options]

After training, set in .env:
    FINE_TUNED_MODEL_PATH=models/pii_gliner
    GLINER_THRESHOLD=0.55
Then restart the server.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_DATA = ROOT / "train" / "data" / "pii_train.json"
DEFAULT_OUT  = ROOT / "models" / "pii_gliner"
DEFAULT_BASE = "urchade/gliner_large-v2.1"

# ── Hardware profiles ─────────────────────────────────────────────────────────

# A100 / H100 / A6000 (>=40 GB VRAM) — max throughput
A100_PROFILE = {
    "epochs":           10,
    "batch_size":       32,
    "grad_accum":        1,    # effective batch = 32
    "lr":             3e-6,
    "warmup_ratio":    0.1,
    "weight_decay":   0.01,
    "fp16":           True,
    "max_grad_norm":   1.0,
    "eval_strategy": "epoch",
    "save_limit":        3,
}

# RTX 3090/4090, A10, V100 (<40 GB VRAM)
GPU_PROFILE = {
    "epochs":           10,
    "batch_size":       16,
    "grad_accum":        1,    # effective batch = 16
    "lr":             5e-6,
    "warmup_ratio":    0.1,
    "weight_decay":   0.01,
    "fp16":           True,    # mixed precision on CUDA
    "max_grad_norm":   1.0,
    "eval_strategy": "epoch",
    "save_limit":        3,
}

# Apple MPS: batch 4 + grad_accum 4 = effective batch 16; no fp16 (MPS doesn't support it)
MPS_PROFILE = {
    "epochs":           10,
    "batch_size":        4,
    "grad_accum":        4,    # effective batch = 16
    "lr":             5e-6,
    "warmup_ratio":    0.1,
    "weight_decay":   0.01,
    "fp16":          False,
    "max_grad_norm":   1.0,
    "eval_strategy": "epoch",
    "save_limit":        3,
}

CPU_PROFILE = {
    "epochs":            5,
    "batch_size":        4,
    "grad_accum":        4,    # effective batch = 16
    "lr":             5e-6,
    "warmup_ratio":    0.1,
    "weight_decay":   0.01,
    "fp16":          False,    # no mixed precision on CPU
    "max_grad_norm":   1.0,
    "eval_strategy": "epoch",
    "save_limit":        2,
}


def _free_ram_gb() -> float:
    """Approximate free system RAM in GB via vm_stat (macOS) or /proc/meminfo."""
    try:
        import subprocess, re
        out = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL).decode()
        free = int(re.search(r"Pages free:\s+(\d+)", out).group(1))
        spec = int(re.search(r"Pages speculative:\s+(\d+)", out).group(1))
        return (free + spec) * 16384 / 1e9
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except Exception:
        pass
    return 999.0  # unknown — assume plenty


def _detect_device() -> tuple[str, dict]:
    """Return (device_str, profile_dict)."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU detected : {name}  ({vram_gb:.1f} GB VRAM)")
            if vram_gb >= 40:  # A100 80GB, H100, A6000 48GB
                print(f"  Profile      : A100/H100 (batch 32, fp16)")
                return "cuda", A100_PROFILE
            return "cuda", GPU_PROFILE
        if torch.backends.mps.is_available():
            free_gb = _free_ram_gb()
            # GLiNER large needs ~14 GB headroom (model + Adam moments + activations).
            # MPS shares unified memory with the OS; fall back to CPU when tight.
            if free_gb >= 14.0:
                print(f"  GPU detected : Apple MPS  ({free_gb:.1f} GB free RAM)")
                return "mps", MPS_PROFILE
            else:
                print(f"  Apple MPS available but only {free_gb:.1f} GB RAM free — using CPU to avoid OOM")
    except Exception:
        pass
    cpu_count = os.cpu_count() or 1
    print(f"  No GPU found — using CPU ({cpu_count} cores)")
    return "cpu", CPU_PROFILE


def load_data(path: Path, seed: int) -> tuple[list[dict], list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * 0.9)
    return data[:split], data[split:]


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{seconds / 60:.1f}min"


def run_training(
    data_path: Path,
    base_model: str,
    out_dir: Path,
    device: str,
    profile: dict,
    seed: int,
) -> None:
    try:
        from gliner import GLiNER
        from gliner.training import Trainer, TrainingArguments
        from gliner.data_processing import UniEncoderSpanDataCollator, UniEncoderTokenDataCollator
        from gliner.data_processing.processor import UniEncoderSpanProcessor, UniEncoderTokenProcessor
    except ImportError:
        print("\nERROR: gliner not installed. Run:\n  pip install gliner torch transformers", file=sys.stderr)
        sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading training data from {data_path}…")
    train_data, val_data = load_data(data_path, seed)
    print(f"  Train : {len(train_data):,} examples")
    print(f"  Val   : {len(val_data):,} examples")

    # ── Count entity coverage ────────────────────────────────────────────────
    entity_counts: dict[str, int] = {}
    for ex in train_data:
        for span in ex.get("ner", []):
            lbl = span[2] if len(span) > 2 else "?"
            entity_counts[lbl] = entity_counts.get(lbl, 0) + 1
    print(f"  Entity types in train : {len(entity_counts)}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading base model: {base_model}")
    t_load = time.perf_counter()
    model = GLiNER.from_pretrained(base_model)
    print(f"  Model loaded in {_fmt_time(time.perf_counter() - t_load)}")

    # ── Compute steps ────────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_data) / (profile["batch_size"] * profile["grad_accum"]))
    total_steps = steps_per_epoch * profile["epochs"]
    warmup_steps = int(total_steps * profile["warmup_ratio"])

    use_cpu = device == "cpu"

    print(f"\nTraining profile ({device.upper()}):")
    print(f"  Epochs          : {profile['epochs']}")
    print(f"  Batch size      : {profile['batch_size']}  (grad accum x{profile['grad_accum']})")
    print(f"  Effective batch : {profile['batch_size'] * profile['grad_accum']}")
    print(f"  Learning rate   : {profile['lr']}")
    print(f"  Total steps     : {total_steps:,}  (warmup {warmup_steps})")
    print(f"  Mixed precision : {'yes' if profile['fp16'] else 'no'}")

    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),

        # Learning rate
        learning_rate=profile["lr"],
        weight_decay=profile["weight_decay"],
        others_lr=profile["lr"] * 10,         # classifier head LR (10x backbone)
        others_weight_decay=profile["weight_decay"],

        # Schedule
        lr_scheduler_type="cosine",
        warmup_ratio=profile["warmup_ratio"],

        # Batch
        per_device_train_batch_size=profile["batch_size"],
        per_device_eval_batch_size=profile["batch_size"] * 2,
        gradient_accumulation_steps=profile["grad_accum"],

        # Training length
        num_train_epochs=profile["epochs"],

        # Eval + checkpoint
        eval_strategy=profile["eval_strategy"],
        save_strategy=profile["eval_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=profile["save_limit"],

        # Precision
        fp16=profile["fp16"] and not use_cpu,

        # Reproducibility
        seed=seed,

        # Misc
        dataloader_num_workers=0,    # avoid multiprocessing issues on macOS/CPU
        use_cpu=use_cpu,
        report_to="none",            # disable wandb / tensorboard unless user sets up
        logging_steps=max(1, steps_per_epoch // 4),
        max_grad_norm=profile["max_grad_norm"],
    )

    # ── Build GLiNER-native data collator ────────────────────────────────────
    tokenizer    = model.data_processor.transformer_tokenizer
    words_split  = model.data_processor.words_splitter
    model_type   = getattr(model.config, "model_type", "gliner_uni_encoder_span")

    if "token" in model_type:
        processor = UniEncoderTokenProcessor(model.config, tokenizer, words_split)
        collator  = UniEncoderTokenDataCollator(model.config, data_processor=processor)
    else:
        processor = UniEncoderSpanProcessor(model.config, tokenizer, words_split)
        collator  = UniEncoderSpanDataCollator(model.config, data_processor=processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
    )

    print(f"\nFine-tuning started…  (estimated: {_fmt_time(total_steps * (0.8 if device == 'cuda' else 4.0))})")
    t_train = time.perf_counter()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving current state…")

    elapsed = time.perf_counter() - t_train
    print(f"\nTraining complete in {_fmt_time(elapsed)}")

    # ── Save final model ─────────────────────────────────────────────────────
    model.save_pretrained(str(out_dir))
    print(f"Model saved to: {out_dir}")

    # ── Write training summary ────────────────────────────────────────────────
    summary = {
        "base_model": base_model,
        "device": device,
        "epochs": profile["epochs"],
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "entity_types": len(entity_counts),
        "elapsed_s": round(elapsed, 1),
        "output_dir": str(out_dir),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Next steps ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print(f"\n1. Set in .env:")
    print(f"     FINE_TUNED_MODEL_PATH={out_dir}")
    print(f"     GLINER_THRESHOLD=0.55")
    print(f"\n2. Restart the server:  ./run.sh")
    print(f"\n3. Test:  python main.py mask-text --text 'Patient John Smith DOB 07/15/1985'")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    device_str, auto_profile = _detect_device()

    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER for PII detection (GPU or CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       type=Path, default=DEFAULT_DATA,
                        help="Training data JSON (output of generate_data.py)")
    parser.add_argument("--base-model", default=DEFAULT_BASE,
                        help="Base GLiNER HuggingFace model to fine-tune")
    parser.add_argument("--out",        type=Path, default=DEFAULT_OUT,
                        help="Output directory for fine-tuned model")
    parser.add_argument("--device",     default=device_str,
                        choices=["cpu", "cuda", "mps"],
                        help="Training device (auto-detected)")
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="Override epoch count from profile")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="Override per-device batch size from profile")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Override learning rate from profile")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    if not args.data.exists():
        print(f"\nERROR: Training data not found at {args.data}", file=sys.stderr)
        print("Run first:  python train/generate_data.py", file=sys.stderr)
        sys.exit(1)

    # Re-detect profile if device was overridden on CLI
    if args.device != device_str:
        if args.device == "cpu":
            profile = CPU_PROFILE.copy()
        elif args.device == "mps":
            profile = MPS_PROFILE.copy()
        else:
            # On CLI override to cuda, still pick A100 profile if the GPU is large
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    profile = A100_PROFILE.copy() if vram_gb >= 40 else GPU_PROFILE.copy()
                else:
                    profile = GPU_PROFILE.copy()
            except Exception:
                profile = GPU_PROFILE.copy()
    else:
        profile = auto_profile.copy()

    # Apply CLI overrides
    if args.epochs is not None:
        profile["epochs"] = args.epochs
    if args.batch_size is not None:
        profile["batch_size"] = args.batch_size
    if args.lr is not None:
        profile["lr"] = args.lr

    run_training(
        data_path=args.data,
        base_model=args.base_model,
        out_dir=args.out,
        device=args.device,
        profile=profile,
        seed=args.seed,
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GLiNER PII Fine-Tuning")
    print("=" * 60)
    print(f"  Entities source: entities_config.yaml")
    print()
    main()
