"""Full-set (824) VoiceBank+DEMAND evaluation with infer_opt optimizations.

Cross-product of {fp32, fp16} x {chunked-2s-overlap0.5, full-utterance},
all with patch(B) + swoosh_fp16_fix applied. Reports WB-PESQ / STOI /
SI-SDR / pesq_failed / OOM count per config, plus dataset-wide peak GPU
mem (reset once at start of each config).

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
      python -u -m zipenhancer_repro.infer_opt.eval_full \
        --config configs/zipenhancer_s.yaml \
        --ckpt   checkpoints/example.pt \
        --configs fp32_chunk fp16_chunk fp16_full

Available configs (--configs):
    fp32_chunk : fp32 + patch(B) + chunked (2s, 0.5 overlap)  [baseline reference]
    fp16_chunk : fp16 + patch(B) + chunked (2s, 0.5 overlap)  [main interest]
    fp32_full  : fp32 + patch(B) + full-utterance forward
    fp16_full  : fp16 + patch(B) + full-utterance forward
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import librosa
import numpy as np
import torch
import yaml

from zipenhancer_repro.models.backbone import build_backbone, mag_pha_stft, mag_pha_istft
from zipenhancer_repro.utils.metrics import si_sdr, compute_pesq, compute_stoi
from zipenhancer_repro.infer_opt.patches import rel_pos_no_repeat, swoosh_fp16_fix


@torch.no_grad()
def _enhance_chunk(g, noisy_chunk, cfg, device, model_dtype, norm):
    s = cfg["stft"]
    x = torch.from_numpy(noisy_chunk).float().unsqueeze(0).to(device)
    if isinstance(norm, float):
        norm_t = torch.tensor(norm, device=device)
    else:
        norm_t = norm
    xin = x * norm_t
    # STFT stays in fp32 (cuFFT doesn't support fp16); model gets cast inputs.
    mag, pha, _ = mag_pha_stft(xin, s["n_fft"], s["hop_length"],
                                s["win_length"], s["compress_factor"])
    out = g(mag.to(model_dtype), pha.to(model_dtype))
    audio = mag_pha_istft(out[0].float(), out[1].float(), s["n_fft"],
                          s["hop_length"], s["win_length"], s["compress_factor"])
    return (audio / norm_t).squeeze(0).cpu().numpy()


def enhance_chunked(g, noisy, cfg, device, model_dtype,
                    chunk_seconds=2.0, overlap=0.5):
    """Overlap-add chunked inference, dtype-aware. Mirrors zipenhancer_repro.infer."""
    sr = cfg["stft"]["sample_rate"]
    chunk_len = int(chunk_seconds * sr)
    hop_len = int(chunk_len * (1.0 - overlap))
    n = len(noisy)
    norm = float(np.sqrt(n / (np.sum(noisy ** 2.0) + 1e-9)))

    if n <= chunk_len:
        return _enhance_chunk(g, noisy, cfg, device, model_dtype, norm)

    window = np.hanning(chunk_len + 1)[:-1]
    pad_head = pad_tail = hop_len
    noisy_padded = np.concatenate([
        noisy[:pad_head][::-1].copy(),
        noisy,
        noisy[-pad_tail:][::-1].copy(),
    ])
    n_padded = len(noisy_padded)
    out = np.zeros(n_padded, dtype=np.float32)
    wsum = np.zeros(n_padded, dtype=np.float32)
    pos = 0
    while pos + chunk_len <= n_padded:
        chunk = noisy_padded[pos:pos + chunk_len]
        enhanced = _enhance_chunk(g, chunk, cfg, device, model_dtype, norm)
        out[pos:pos + chunk_len] += enhanced * window
        wsum[pos:pos + chunk_len] += window
        pos += hop_len
    wsum[wsum < 1e-8] = 1.0
    return (out / wsum)[pad_head:pad_head + n]


@torch.no_grad()
def enhance_full(g, noisy, cfg, device, model_dtype):
    """Single-shot full-utterance forward."""
    s = cfg["stft"]
    x = torch.from_numpy(noisy).float().unsqueeze(0).to(device)
    norm = torch.sqrt(x.shape[1] / (torch.sum(x ** 2.0) + 1e-9))
    xin = x * norm
    mag, pha, _ = mag_pha_stft(xin, s["n_fft"], s["hop_length"],
                                s["win_length"], s["compress_factor"])
    out = g(mag.to(model_dtype), pha.to(model_dtype))
    audio = mag_pha_istft(out[0].float(), out[1].float(), s["n_fft"],
                          s["hop_length"], s["win_length"], s["compress_factor"])
    return (audio / norm).squeeze(0).cpu().numpy()


def run_eval(g, cfg, device, model_dtype, names, mode):
    """mode: 'chunk' or 'full'. Returns dict with metrics + peak + OOM count."""
    sr = cfg["stft"]["sample_rate"]
    noisy_dir = cfg["data"]["voicebank"]["test_noisy"]
    clean_dir = cfg["data"]["voicebank"]["test_clean"]
    g.eval()
    torch.cuda.reset_peak_memory_stats(device)
    pesqs, stois, sisdrs = [], [], []
    failed_pesq = 0
    oom = 0
    t0 = time.time()
    for i, nm in enumerate(names):
        noisy, _ = librosa.load(os.path.join(noisy_dir, nm), sr=sr)
        clean, _ = librosa.load(os.path.join(clean_dir, nm), sr=sr)
        try:
            if mode == "chunk":
                enh = enhance_chunked(g, noisy, cfg, device, model_dtype,
                                      chunk_seconds=2.0, overlap=0.5)
            else:
                enh = enhance_full(g, noisy, cfg, device, model_dtype)
        except torch.cuda.OutOfMemoryError:
            oom += 1
            gc.collect()
            torch.cuda.empty_cache()
            continue
        n = min(len(clean), len(enh))
        clean, enh = clean[:n], enh[:n]
        p = compute_pesq(clean, enh, sr, "wb")
        if p != p:
            failed_pesq += 1
        else:
            pesqs.append(p)
            stois.append(compute_stoi(clean, enh, sr))
            sisdrs.append(si_sdr(enh, clean))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(names)}] PESQ_running="
                  f"{np.mean(pesqs):.4f}  oom={oom}", flush=True)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return {
        "n_total": len(names),
        "n_pesq": len(pesqs),
        "pesq_failed": failed_pesq,
        "oom": oom,
        "wb_pesq": float(np.mean(pesqs)) if pesqs else float("nan"),
        "stoi": float(np.mean(stois)) if stois else float("nan"),
        "si_sdr": float(np.mean(sisdrs)) if sisdrs else float("nan"),
        "peak_mb": peak_mb,
        "elapsed_s": time.time() - t0,
    }


def build_g(ckpt_path, dtype, device):
    g = build_backbone().to(device).eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck
    g.load_state_dict(state, strict=True)
    if dtype == torch.float16:
        g = g.half()
    return g, ck


CONFIGS = {
    "fp32_chunk": (torch.float32, "chunk"),
    "fp16_chunk": (torch.float16, "chunk"),
    "fp32_full":  (torch.float32, "full"),
    "fp16_full":  (torch.float16, "full"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", type=int, default=None,
                    help="limit to first N utterances (None = full 824)")
    ap.add_argument("--configs", nargs="+", default=["fp32_chunk", "fp16_chunk"],
                    choices=list(CONFIGS.keys()))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    assert torch.cuda.is_available()
    device = "cuda"

    # Activate patches once. patch(B) is bit-exact for fp32, dtype-cast helper
    # for fp16; swoosh_fp16_fix is no-op for fp32.
    rel_pos_no_repeat.apply()
    swoosh_fp16_fix.apply()
    print(f"[patches] rel_pos_no_repeat={rel_pos_no_repeat.is_active()}  "
          f"swoosh_fp16_fix={swoosh_fp16_fix.is_active()}")

    noisy_dir = cfg["data"]["voicebank"]["test_noisy"]
    names = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))
    if args.subset:
        names = names[:args.subset]
    print(f"[data] {len(names)} utterances")

    ck_info = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    step = ck_info.get("step", "?") if isinstance(ck_info, dict) else "?"
    best = ck_info.get("best_pesq", float("nan")) if isinstance(ck_info, dict) else float("nan")
    print(f"[ckpt] step={step}  best_pesq={best:.4f}")
    del ck_info

    results = {}
    for tag in args.configs:
        dtype, mode = CONFIGS[tag]
        print(f"\n>>> {tag}  dtype={dtype}  mode={mode}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        g, _ = build_g(args.ckpt, dtype, device)
        r = run_eval(g, cfg, device, dtype, names, mode)
        results[tag] = r
        print(f"  n={r['n_pesq']}/{r['n_total']}  oom={r['oom']}  "
              f"pesq_failed={r['pesq_failed']}  | "
              f"PESQ={r['wb_pesq']:.4f}  STOI={r['stoi']:.4f}  "
              f"SI-SDR={r['si_sdr']:.3f}  | peak={r['peak_mb']:.1f}MB  "
              f"elapsed={r['elapsed_s']:.0f}s", flush=True)
        del g
        gc.collect()
        torch.cuda.empty_cache()

    print("\n========== MARKDOWN TABLE ==========")
    print("| config | n_pesq/total | OOM | WB-PESQ | STOI | SI-SDR | peak(MB) | elapsed(s) |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, r in results.items():
        print(f"| {tag} | {r['n_pesq']}/{r['n_total']} | {r['oom']} | "
              f"{r['wb_pesq']:.4f} | {r['stoi']:.4f} | {r['si_sdr']:.3f} | "
              f"{r['peak_mb']:.1f} | {r['elapsed_s']:.0f} |")


if __name__ == "__main__":
    main()
