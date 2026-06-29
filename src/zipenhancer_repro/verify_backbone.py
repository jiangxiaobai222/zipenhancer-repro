"""Strict-load verification for the packaged official-compatible backbone."""
from __future__ import annotations

import argparse

import torch

from .models.backbone import build_backbone


def _generator_state(ck):
    return ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="official pytorch_model.bin or generator checkpoint")
    ap.add_argument("--config", default=None, help="optional official-style configuration.json")
    args = ap.parse_args()

    model = build_backbone(args.config)
    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(_generator_state(ck), strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] strict-compatible load succeeded. params={n_params/1e6:.3f}M")


if __name__ == "__main__":
    main()
