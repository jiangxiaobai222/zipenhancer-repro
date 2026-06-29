"""Peak-memory comparison: original rel-pos vs patch (B).

For each utterance length (seconds), do a single full-utterance forward
(no chunking) and record `torch.cuda.max_memory_allocated`. Compare
patch-OFF vs patch-ON. Tries lengths until OOM, then continues with the
next length using only the patched path.

Output:
    A markdown table summarizing (secs, T, orig_peak_MB, patched_peak_MB,
    speedup, OOM flags).
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
import yaml

from zipenhancer_repro.models.backbone import build_backbone, mag_pha_stft, mag_pha_istft
from zipenhancer_repro.infer_opt.patches import rel_pos_no_repeat


@torch.no_grad()
def _forward(g, noisy, cfg):
    s = cfg["stft"]
    norm = torch.sqrt(noisy.shape[1] / (torch.sum(noisy ** 2.0) + 1e-9))
    xin = noisy * norm
    mag, pha, _ = mag_pha_stft(xin, s["n_fft"], s["hop_length"],
                                s["win_length"], s["compress_factor"])
    out = g(mag, pha)
    audio_g = mag_pha_istft(out[0], out[1], s["n_fft"], s["hop_length"],
                             s["win_length"], s["compress_factor"])
    return audio_g / norm


def _measure(g, noisy, cfg, device):
    """Returns (peak_MB, elapsed_s) or (None, None) on OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        t0 = time.time()
        _ = _forward(g, noisy, cfg)
        torch.cuda.synchronize()
        dt = time.time() - t0
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024, dt
    except torch.cuda.OutOfMemoryError:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sr = cfg["stft"]["sample_rate"]
    assert torch.cuda.is_available(), "GPU required"
    device = "cuda"

    print(f"[setup] device={device}  ckpt={args.ckpt}")
    g = build_backbone().to(device).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck
    g.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in g.parameters())
    weight_mb = sum(p.numel() * p.element_size()
                    for p in g.parameters()) / 1024 / 1024
    print(f"[setup] params={n_params/1e6:.3f}M  weight={weight_mb:.2f}MB")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = []
    for secs in args.seconds:
        L = int(secs * sr)
        noisy = ((torch.randn(1, L, generator=gen) * 0.1)
                 .clamp(-0.99, 0.99).to(device))
        T_frames = L // cfg["stft"]["hop_length"] + 1

        # ORIGINAL
        rel_pos_no_repeat.revert()
        peak_o, dt_o = _measure(g, noisy, cfg, device)
        # PATCHED
        rel_pos_no_repeat.apply()
        peak_p, dt_p = _measure(g, noisy, cfg, device)
        rel_pos_no_repeat.revert()

        def fmt_mb(x): return "OOM" if x is None else f"{x:.1f}"
        def fmt_s(x): return "-" if x is None else f"{x:.2f}"
        ratio = "-" if (peak_o is None or peak_p is None) else f"{peak_o/peak_p:.2f}x"
        speedup = "-" if (dt_o is None or dt_p is None) else f"{dt_o/dt_p:.2f}x"
        rows.append((secs, T_frames, peak_o, peak_p, dt_o, dt_p))
        print(f"  secs={secs:>5.1f}  T={T_frames:>5}  "
              f"orig={fmt_mb(peak_o):>7}MB ({fmt_s(dt_o)}s)  "
              f"pat={fmt_mb(peak_p):>7}MB ({fmt_s(dt_p)}s)  "
              f"mem_ratio={ratio}  speed_ratio={speedup}",
              flush=True)
        del noisy

    # Markdown table
    print("\n========== MARKDOWN TABLE ==========")
    print("| secs | T(帧) | orig_peak(MB) | patched_peak(MB) | mem_ratio | orig_time(s) | patched_time(s) |")
    print("|---|---|---|---|---|---|---|")
    for secs, T, po, pp, dto, dtp in rows:
        po_s = "**OOM**" if po is None else f"{po:.1f}"
        pp_s = "**OOM**" if pp is None else f"{pp:.1f}"
        ratio = "-" if (po is None or pp is None) else f"**{po/pp:.2f}x**"
        dto_s = "-" if dto is None else f"{dto:.2f}"
        dtp_s = "-" if dtp is None else f"{dtp:.2f}"
        print(f"| {secs:.1f} | {T} | {po_s} | {pp_s} | {ratio} | {dto_s} | {dtp_s} |")

    print("\n[done]")


if __name__ == "__main__":
    main()
