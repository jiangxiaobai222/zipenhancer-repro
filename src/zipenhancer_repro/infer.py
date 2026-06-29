"""Practical inference: enhance wav file(s) with overlap-add chunking.

The model is trained on 2-second segments and uses global self-attention
(O(T^2) memory), so feeding a long utterance directly will OOM.  This script
splits the input into 2-second chunks with 50% overlap, enhances each chunk
on GPU (small T -> small memory), and recombines via a Hann window
(overlap-add) to produce a seamless output.

Usage:
    # single file
    python -m zipenhancer_repro.infer --config configs/zipenhancer_s.yaml \
        --ckpt checkpoints/example.pt --input noisy.wav --output enhanced.wav

    # batch (folder)
    python -m zipenhancer_repro.infer --config ... --ckpt ... \
        --input noisy_dir/ --output enhanced_dir/

    # force CPU (slow but no GPU needed)
    python -m zipenhancer_repro.infer --config ... --ckpt ... --input ... --output ... --device cpu
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torchaudio

from .models.backbone import build_backbone, mag_pha_stft, mag_pha_istft


def _generator_state(ck):
    return ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck


def _load(path, sr):
    wav, in_sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)      # mono
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    return wav.squeeze(0).numpy()            # 1-D np.array


@torch.no_grad()
def _enhance_chunk(generator, noisy_chunk, cfg, device, norm=None):
    """Enhance a short chunk (<= training segment length).  Mirrors the official
    norm pipeline: norm -> STFT -> forward -> iSTFT -> denorm.

    norm: if provided (for chunked inference), use the GLOBAL norm factor of
    the full utterance so that all chunks share the same scale — per-chunk norm
    causes amplitude discontinuities at block boundaries and hurts PESQ.
    """
    s = cfg["stft"]
    n_fft, hop, win, cf = s["n_fft"], s["hop_length"], s["win_length"], s["compress_factor"]
    center = s.get("center", True)
    x = torch.from_numpy(noisy_chunk).float().unsqueeze(0).to(device)
    if norm is None:
        norm = torch.sqrt(x.shape[1] / (torch.sum(x ** 2.0) + 1e-9))
    else:
        norm = torch.tensor(norm, device=device)
    xin = x * norm
    mag, pha, _ = mag_pha_stft(xin, n_fft, hop, win, cf, center=center)
    out = generator(mag, pha)
    audio = mag_pha_istft(out[0], out[1], n_fft, hop, win, cf, center=center)
    return (audio / norm).squeeze(0).cpu().numpy()


def enhance_long(generator, noisy, cfg, device, chunk_seconds=2.0, overlap=0.5):
    """Enhance an arbitrarily long 1-D numpy array via overlap-add chunking.

    chunk_seconds: match training segment length (2s) for best fidelity.
    overlap: 0.5 -> 50% overlap, Hann crossfade at boundaries.

    Key: the norm factor is computed ONCE on the full utterance (not per-chunk)
    to avoid amplitude discontinuities at block boundaries.
    """
    sr = cfg["stft"]["sample_rate"]
    chunk_len = int(chunk_seconds * sr)
    hop_len = int(chunk_len * (1.0 - overlap))
    n = len(noisy)

    # Global norm factor (same as enhance_wav, but computed on the full signal).
    norm = float(np.sqrt(n / (np.sum(noisy ** 2.0) + 1e-9)))

    if n <= chunk_len:
        return _enhance_chunk(generator, noisy, cfg, device, norm=norm)

    # Hann window for smooth crossfade (same family as STFT window).
    window = np.hanning(chunk_len + 1)[:-1]

    # Reflect-pad head AND tail by hop_len so that the first/last samples fall
    # in the window CENTER (not edge where Hann=0), preventing signal loss at
    # the utterance boundaries.
    pad_head = hop_len
    pad_tail = hop_len
    noisy_padded = np.concatenate([
        noisy[:pad_head][::-1].copy(),   # reflect (not zero) to avoid silence
        noisy,
        noisy[-pad_tail:][::-1].copy(),
    ])
    n_padded = len(noisy_padded)

    out = np.zeros(n_padded, dtype=np.float32)
    wsum = np.zeros(n_padded, dtype=np.float32)

    pos = 0
    while pos + chunk_len <= n_padded:
        chunk = noisy_padded[pos:pos + chunk_len]
        enhanced = _enhance_chunk(generator, chunk, cfg, device, norm=norm)
        out[pos:pos + chunk_len] += enhanced * window
        wsum[pos:pos + chunk_len] += window
        pos += hop_len

    # Normalize by window sum (avoid division by zero at edges).
    wsum[wsum < 1e-8] = 1.0
    out = out / wsum
    # Strip the reflect-padding to return exactly n samples.
    return out[pad_head:pad_head + n]


def main():
    ap = argparse.ArgumentParser(
        description="Practical ZipEnhancer inference with overlap-add chunking")
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint (best.pt / latest.pt / ckpt_*.pt)")
    ap.add_argument("--input", required=True, help="wav file or folder")
    ap.add_argument("--output", required=True, help="wav file or folder")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--chunk_seconds", type=float, default=2.0,
                    help="chunk length in seconds (match training segment)")
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="overlap ratio 0..1 (0.5 = 50%)")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    sr = cfg["stft"]["sample_rate"]

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = build_backbone().to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    generator.load_state_dict(_generator_state(ck), strict=True)
    generator.eval()
    step = ck.get("step", "?") if isinstance(ck, dict) else "?"
    print(f"[inference] loaded {args.ckpt} (step={step}) | device={device} "
          f"| chunk={args.chunk_seconds}s overlap={args.overlap}")

    # Collect input/output pairs.
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
        enhanced = enhance_long(generator, noisy, cfg, device,
                                chunk_seconds=args.chunk_seconds,
                                overlap=args.overlap)
        torchaudio.save(dst, torch.from_numpy(enhanced).unsqueeze(0).float(), sr)
        print(f"[ok] {os.path.basename(src)} ({len(noisy)/sr:.1f}s) -> {dst}")


if __name__ == "__main__":
    main()
