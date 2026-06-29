#!/usr/bin/env bash
set -euo pipefail

python -m zipenhancer_repro.evaluate \
  --config configs/zipenhancer_s.yaml \
  --ckpt checkpoints/example.pt \
  --subset 100
