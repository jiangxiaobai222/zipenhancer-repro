#!/usr/bin/env bash
set -euo pipefail

python -m zipenhancer_repro.train \
  --config configs/zipenhancer_s.yaml \
  --smoke
