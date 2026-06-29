#!/usr/bin/env bash
set -euo pipefail

python -m zipenhancer_repro.infer \
  --config configs/zipenhancer_s.yaml \
  --ckpt checkpoints/example.pt \
  --input data/noisy_dir \
  --output outputs/enhanced_dir
