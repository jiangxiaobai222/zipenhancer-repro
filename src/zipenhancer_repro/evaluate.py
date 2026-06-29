"""Evaluation on the VoiceBank test set: WB-PESQ / STOI / SI-SDR.

Mirrors the official inference normalization (norm_factor = sqrt(L/sum(noisy^2))).
No silent fallbacks: PESQ failures are COUNTED and reported explicitly (not hidden
in a nan-average), per project rule.

Usage:
    python -m zipenhancer_repro.evaluate --config configs/zipenhancer_s.yaml --ckpt checkpoints/example.pt
    python -m zipenhancer_repro.evaluate --config ... --ckpt ... --subset 100
"""
from __future__ import annotations

import argparse
import os

import librosa
import numpy as np
import torch

from .models.backbone import build_backbone
from .infer import enhance_long
from .utils.metrics import si_sdr, compute_pesq, compute_stoi


@torch.no_grad()
def enhance_wav(generator, noisy, cfg, device):
    """noisy: 1-D np.array @16k -> enhanced 1-D np.array.

    Uses the same chunked overlap-add path as `zipenhancer_repro.infer` (2s chunks,
    50% Hann overlap, GLOBAL norm factor of the full utterance).  Rationale:
    the Zipformer dual-path attention + icefall rel-pos gather is O(heads*F*T^2)
    in memory (freq folded into the attention batch dim), so a full-utterance
    forward can OOM on long test files. Chunking locks T to the 2s
    training length (T~321 -> peak ~1GB, constant w.r.t. utterance length) and
    keeps a full 2s context per chunk, so metrics are essentially unchanged.
    """
    return enhance_long(generator, noisy, cfg, device,
                        chunk_seconds=2.0, overlap=0.5)


def evaluate_generator(generator, cfg, device, subset=None, verbose=True,
                       tb_sample=False, fixed_seed=1234):
    """Evaluate on the VoiceBank test set.

    When `tb_sample=True`, ALSO collect TWO test utterances for TensorBoard:
      - "fixed":  a deterministic index from `fixed_seed` -> SAME utterance every
                  eval cycle, so you can listen to one sentence evolve across steps.
      - "random": a fresh random index each call -> rotates, for broader coverage.
    Both REUSE the enhanced waveform already produced by the eval loop (zero extra
    inference). The two are kept distinct when N>1. Results land in res["samples"]
    as {"fixed": {...}, "random": {...}} (each: name/sr/noisy/clean/enhanced/pesq).
    """
    d = cfg["data"]["voicebank"]
    sr = cfg["stft"]["sample_rate"]
    noisy_dir, clean_dir = d["test_noisy"], d["test_clean"]
    names = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))
    if subset:
        names = names[:subset]

    sample_targets = {}  # index -> tag ("fixed"/"random")
    if tb_sample and names:
        N = len(names)
        fixed_idx = int(np.random.default_rng(fixed_seed).integers(N))
        rand_idx = int(np.random.default_rng().integers(N))
        while N > 1 and rand_idx == fixed_idx:
            rand_idx = int(np.random.default_rng().integers(N))
        sample_targets[fixed_idx] = "fixed"
        sample_targets.setdefault(rand_idx, "random")  # N==1 -> only "fixed"
    samples = {}

    generator.eval()
    pesqs, stois, sisdrs, failed = [], [], [], 0
    for i, name in enumerate(names):
        noisy, _ = librosa.load(os.path.join(noisy_dir, name), sr=sr)
        clean, _ = librosa.load(os.path.join(clean_dir, name), sr=sr)
        enh = enhance_wav(generator, noisy, cfg, device)
        if verbose and i > 0 and i % 100 == 0:
            print(f"[eval] progress {i}/{len(names)}", flush=True)
        n = min(len(clean), len(enh))
        clean, enh = clean[:n], enh[:n]

        p = compute_pesq(clean, enh, sr, "wb")
        if p != p:            # nan == PESQ failed on this utterance: surface it
            failed += 1
        else:
            pesqs.append(p)
        stois.append(compute_stoi(clean, enh, sr))
        sisdrs.append(si_sdr(enh, clean))

        if i in sample_targets:
            samples[sample_targets[i]] = {
                "name": name, "sr": sr,
                "noisy": noisy[:n].astype(np.float32),
                "clean": clean.astype(np.float32),
                "enhanced": enh.astype(np.float32),
                "pesq": float(p),  # may be nan -> surfaced as-is, not hidden
            }

    res = {
        "wb_pesq": float(np.mean(pesqs)) if pesqs else float("nan"),
        "stoi": float(np.mean(stois)),
        "si_sdr": float(np.mean(sisdrs)),
        "n": len(names),
        "pesq_failed": failed,
    }
    if tb_sample:
        res["samples"] = samples
    if verbose:
        print(f"[eval] n={res['n']} pesq_failed={failed} | "
              f"WB-PESQ {res['wb_pesq']:.4f} | STOI {res['stoi']:.4f} | "
              f"SI-SDR {res['si_sdr']:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = build_backbone().to(device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    generator.load_state_dict(ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck, strict=True)
    step = ck.get("step", "?") if isinstance(ck, dict) else "?"
    print(f"[eval] loaded {args.ckpt} (step={step})")
    evaluate_generator(generator, cfg, device, subset=args.subset)


if __name__ == "__main__":
    main()
