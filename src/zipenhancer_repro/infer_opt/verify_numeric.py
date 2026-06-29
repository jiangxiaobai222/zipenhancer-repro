"""Numerical equivalence: original rel-pos gather vs optimized gather.

Builds two generators with identical state_dict, feeds the same noisy input
through both -- one with the patch active, one
without — and reports max-abs-diff for (mag_g, pha_g, audio_g).

PASS criterion: max diff <= 1e-5 in fp32 (numerical noise of float32
matmul / addition).
"""
from __future__ import annotations

import argparse
import copy

import numpy as np
import torch

from zipenhancer_repro.models.backbone import build_backbone, mag_pha_stft, mag_pha_istft
from zipenhancer_repro.infer_opt.patches import rel_pos_no_repeat


@torch.no_grad()
def _forward(generator, noisy: torch.Tensor, cfg) -> tuple:
    """Returns (mag_g, pha_g, audio_g) for given (1, L) noisy input."""
    s = cfg["stft"]
    norm = torch.sqrt(noisy.shape[1] / (torch.sum(noisy ** 2.0) + 1e-9))
    xin = noisy * norm
    mag, pha, _ = mag_pha_stft(xin, s["n_fft"], s["hop_length"],
                                s["win_length"], s["compress_factor"])
    out = generator(mag, pha)
    mag_g, pha_g = out[0], out[1]
    audio_g = mag_pha_istft(mag_g, pha_g, s["n_fft"], s["hop_length"],
                            s["win_length"], s["compress_factor"])
    return mag_g, pha_g, audio_g / norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    sr = cfg["stft"]["sample_rate"]
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}  tol={args.tol}")

    # Build TWO generators with same weights so eval-time stochasticity (none,
    # since dropout/pos_emb_skip are inactive in .eval()) cannot confound.
    print("[setup] building two generators with identical state_dict ...")
    g_orig = build_backbone().to(device).eval()
    g_pat = build_backbone().to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck
    g_orig.load_state_dict(state, strict=True)
    g_pat.load_state_dict(state, strict=True)
    step = ck.get("step", "?") if isinstance(ck, dict) else "?"
    best = ck.get("best_pesq", float("nan")) if isinstance(ck, dict) else float("nan")
    print(f"[setup] loaded ckpt: step={step}  best_pesq={best:.4f}")

    # Sanity: patch is OFF at start.
    assert not rel_pos_no_repeat.is_active()

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    all_pass = True
    for secs in args.seconds:
        L = int(secs * sr)
        noisy_cpu = (torch.randn(1, L, generator=g) * 0.1).clamp(-0.99, 0.99)
        noisy = noisy_cpu.to(device)

        # Original forward (patch OFF)
        rel_pos_no_repeat.revert()
        assert not rel_pos_no_repeat.is_active()
        mag_o, pha_o, aud_o = _forward(g_orig, noisy, cfg)

        # Patched forward (patch ON, same input, same weights)
        rel_pos_no_repeat.apply()
        assert rel_pos_no_repeat.is_active()
        mag_p, pha_p, aud_p = _forward(g_pat, noisy, cfg)
        rel_pos_no_repeat.revert()

        d_mag = (mag_o - mag_p).abs().max().item()
        d_pha = (pha_o - pha_p).abs().max().item()
        d_aud = (aud_o - aud_p).abs().max().item()
        max_d = max(d_mag, d_pha, d_aud)
        ok = max_d <= args.tol
        all_pass = all_pass and ok
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] secs={secs:.1f}s L={L}  "
              f"|Δmag|max={d_mag:.2e}  |Δpha|max={d_pha:.2e}  "
              f"|Δaud|max={d_aud:.2e}  (tol={args.tol:.0e})")

    print(f"\n[overall] {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
