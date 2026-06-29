#!/usr/bin/env bash
set -euo pipefail

python -m zipenhancer_repro.infer_opt.verify_numeric \
  --config configs/zipenhancer_s.yaml \
  --ckpt checkpoints/example.pt

python -m zipenhancer_repro.infer_opt.infer_lite \
  --config configs/zipenhancer_s.yaml \
  --ckpt checkpoints/example.pt \
  --input examples/noisy.wav \
  --output outputs/enhanced_lite.wav
