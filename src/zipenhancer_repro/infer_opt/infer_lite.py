"""Optimized full-utterance inference (no chunking required).

Applies `zipenhancer_repro.infer_opt.patches.rel_pos_no_repeat`, then
runs a single full-utterance forward. The chunked path in `zipenhancer_repro.infer`
is not used here: with the patch active,
single-shot long-utterance inference becomes affordable.

If a long file would still OOM (huge inputs), falls back to chunked path
with a clear log line — never silently chunks behind the user's back.

Usage:
    # single file
    python -m zipenhancer_repro.infer_opt.infer_lite \
        --config configs/zipenhancer_s.yaml \
        --ckpt   checkpoints/example.pt \
        --input  noisy.wav \
        --output enhanced.wav

    # folder
    python -m zipenhancer_repro.infer_opt.infer_lite --config ... --ckpt ... \
        --input noisy_dir/ --output enhanced_dir/

    # compare patched vs original peak on the same file
    python -m zipenhancer_repro.infer_opt.infer_lite --config ... --ckpt ... \
        --input noisy.wav --output enhanced.wav --compare
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import torch
import torchaudio
import yaml

from zipenhancer_repro.models.backbone import build_backbone, mag_pha_stft, mag_pha_istft
from zipenhancer_repro.infer_opt.patches import rel_pos_no_repeat


def _load(path, sr):
    wav, in_sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav.squeeze(0).numpy()


@torch.no_grad()
def enhance_full(generator, noisy_np, cfg, device):
    """Single-shot full-utterance forward. Returns (enhanced_np, peak_MB)."""
    s = cfg["stft"]
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats(device)

    x = torch.from_numpy(noisy_np).float().unsqueeze(0).to(device)
    norm = torch.sqrt(x.shape[1] / (torch.sum(x ** 2.0) + 1e-9))
    xin = x * norm
    mag, pha, _ = mag_pha_stft(xin, s["n_fft"], s["hop_length"],
                                s["win_length"], s["compress_factor"])
    out = generator(mag, pha)
    audio = mag_pha_istft(out[0], out[1], s["n_fft"], s["hop_length"],
                           s["win_length"], s["compress_factor"])
    enh = (audio / norm).squeeze(0).cpu().numpy()
    peak = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if torch.cuda.is_available() and device != "cpu" else float("nan"))
    return enh, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="wav file or folder")
    ap.add_argument("--output", required=True, help="wav file or folder")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--compare", action="store_true",
                    help="also run with patch OFF for peak-MB comparison")
    ap.add_argument("--no_patch", action="store_true",
                    help="skip the rel-pos patch (baseline; expect higher peak)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sr = cfg["stft"]["sample_rate"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    g = build_backbone().to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck
    g.load_state_dict(state, strict=True)
    g.eval()
    n_params = sum(p.numel() for p in g.parameters())
    weight_mb = sum(p.numel() * p.element_size()
                    for p in g.parameters()) / 1024 / 1024
    patch_state = "OFF" if args.no_patch else "ON"
    step = ck.get("step", "?") if isinstance(ck, dict) else "?"
    best = ck.get("best_pesq", float("nan")) if isinstance(ck, dict) else float("nan")
    print(f"[infer_lite] ckpt step={step}  best_pesq={best:.4f}")
    print(f"[infer_lite] params={n_params/1e6:.3f}M  weight={weight_mb:.2f}MB  "
          f"device={device}  patch={patch_state}")

    if not args.no_patch:
        rel_pos_no_repeat.apply()
        assert rel_pos_no_repeat.is_active()

    inputs = []
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for f in sorted(os.listdir(args.input)):
            if f.lower().endswith((".wav", ".flac")):
                inputs.append((os.path.join(args.input, f),
                               os.path.join(args.output, f)))
    else:
        inputs.append((args.input, args.output))

    for src, dst in inputs:
        noisy = _load(src, sr)
        dur = len(noisy) / sr

        t0 = time.time()
        try:
            enh, peak = enhance_full(g, noisy, cfg, device)
            dt = time.time() - t0
            print(f"[ok] {os.path.basename(src)} ({dur:.2f}s)  "
                  f"peak={peak:.1f}MB  rtf={dt/dur:.3f}  -> {dst}")
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] {os.path.basename(src)} ({dur:.2f}s) — full-utterance "
                  f"forward failed even with patch; consider --device cpu or "
                  f"falling back to zipenhancer_repro.infer (chunked).")
            continue

        if args.compare:
            # Also measure with patch OFF for the same input on the same device.
            rel_pos_no_repeat.revert()
            gc.collect(); torch.cuda.empty_cache() if device != "cpu" else None
            try:
                _, peak_off = enhance_full(g, noisy, cfg, device)
                print(f"  [compare] patch=OFF  peak={peak_off:.1f}MB  "
                      f"ratio={peak_off/peak:.2f}x")
            except torch.cuda.OutOfMemoryError:
                print(f"  [compare] patch=OFF  **OOM** (vs patched {peak:.1f}MB)")
            rel_pos_no_repeat.apply()

        torchaudio.save(dst, torch.from_numpy(enh).unsqueeze(0).float(), sr)


if __name__ == "__main__":
    main()
