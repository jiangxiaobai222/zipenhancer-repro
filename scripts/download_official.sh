#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-checkpoints/official}"
MODEL_ID="${MODEL_ID:-iic/speech_zipenhancer_ans_multiloss_16k_base}"

mkdir -p "$DEST"

python - "$DEST" "$MODEL_ID" <<'PY'
import os
import shutil
import sys

try:
    from modelscope import snapshot_download
except ImportError as exc:
    raise SystemExit("Install ModelScope first: pip install modelscope") from exc

dest, model_id = sys.argv[1], sys.argv[2]
print(f"[download] {model_id}")
cache_dir = snapshot_download(model_id)
print(f"[cache] {cache_dir}")

for name in os.listdir(cache_dir):
    src = os.path.join(cache_dir, name)
    dst = os.path.join(dest, name)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

print(f"[ok] mirrored files into {dest}")
print("files:", sorted(os.listdir(dest)))
PY
