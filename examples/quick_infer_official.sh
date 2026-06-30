#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-examples/test_datas/speech_with_noise.wav}"
OUTPUT="${2:-outputs/enhanced_official.wav}"

mkdir -p "$(dirname "$OUTPUT")"

set +e
python -m zipenhancer_repro.infer_opt.infer_lite \
  --config configs/zipenhancer_s.yaml \
  --ckpt weights/pytorch_model.bin \
  --input "$INPUT" \
  --output "$OUTPUT"
STATUS=$?
set -e

NEED_FALLBACK=0
if [ "$STATUS" -ne 0 ]; then
  NEED_FALLBACK=1
elif [ -d "$INPUT" ]; then
  IN_COUNT=$(find "$INPUT" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.flac' \) | wc -l)
  OUT_COUNT=$(find "$OUTPUT" -maxdepth 1 -type f \( -iname '*.wav' -o -iname '*.flac' \) 2>/dev/null | wc -l)
  [ "$OUT_COUNT" -lt "$IN_COUNT" ] && NEED_FALLBACK=1
elif [ ! -s "$OUTPUT" ]; then
  NEED_FALLBACK=1
fi

if [ "$NEED_FALLBACK" -eq 1 ]; then
  echo "[fallback] optimized full-utterance inference failed; using chunked overlap-add."
  python -m zipenhancer_repro.infer \
    --config configs/zipenhancer_s.yaml \
    --ckpt weights/pytorch_model.bin \
    --input "$INPUT" \
    --output "$OUTPUT"
fi
