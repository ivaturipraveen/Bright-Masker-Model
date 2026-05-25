"""Download fine-tuned GLiNER weights from S3 to models/pii_gliner/.

Reads AWS credentials and bucket info from .env. Mirrors the logic in run.sh.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")

    bucket = os.getenv("S3_BUCKET", "brightmasker")
    prefix = os.getenv("S3_MODEL_PREFIX", "models/pii_gliner").strip("/")
    out_dir = repo_root / os.getenv("FINE_TUNED_MODEL_PATH", "models/pii_gliner")
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    files = ["pytorch_model.bin", "gliner_config.json", "tokenizer.json", "tokenizer_config.json"]

    out_dir.mkdir(parents=True, exist_ok=True)

    boto_cfg = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "adaptive"},
        connect_timeout=30,
        read_timeout=120,
        max_pool_connections=20,
    )
    s3 = boto3.client("s3", config=boto_cfg)

    transfer_cfg = TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    print(f"Bucket : s3://{bucket}/{prefix}/")
    print(f"Output : {out_dir}\n")

    for name in files:
        key = f"{prefix}/{name}"
        dest = out_dir / name
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            total = head["ContentLength"]
        except (BotoCoreError, ClientError) as exc:
            print(f"  {name:25} SKIP ({exc})")
            continue

        if dest.exists() and dest.stat().st_size == total:
            print(f"  {name:25} already present ({_human_size(total)})")
            continue

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            downloaded = 0
            last_print = 0.0

            def _cb(chunk: int) -> None:
                nonlocal downloaded, last_print
                downloaded += chunk
                now = time.perf_counter()
                if now - last_print > 0.5 or downloaded == total:
                    pct = 100 * downloaded / total
                    rate = downloaded / max(now - t0, 1e-6) / (1024 * 1024)
                    print(
                        f"  {name:25} {pct:5.1f}%  ({_human_size(downloaded)} / {_human_size(total)})  {rate:5.1f} MB/s",
                        end="\r",
                    )
                    last_print = now

            try:
                s3.download_file(bucket, key, str(dest), Callback=_cb, Config=transfer_cfg)
                elapsed = time.perf_counter() - t0
                print(
                    f"  {name:25} ok    ({_human_size(total)} in {elapsed:.1f}s)" + " " * 30
                )
                break
            except (BotoCoreError, ClientError, OSError) as exc:
                print(
                    f"\n  {name:25} attempt {attempt}/{max_attempts} failed: {exc}"
                )
                if attempt == max_attempts:
                    print(f"  {name:25} FAIL after {max_attempts} attempts")
                    return 1
                time.sleep(min(2 ** attempt, 30))

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
