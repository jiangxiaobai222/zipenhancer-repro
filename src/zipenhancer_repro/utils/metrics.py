"""Objective metrics for speech enhancement: PESQ (WB/NB), STOI, SI-SDR."""
from __future__ import annotations

import numpy as np


def si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def compute_pesq(ref: np.ndarray, est: np.ndarray, sr: int = 16000, mode: str = "wb") -> float:
    try:
        from pesq import pesq
        return float(pesq(sr, ref, est, mode))
    except Exception:
        return float("nan")


def compute_stoi(ref: np.ndarray, est: np.ndarray, sr: int = 16000) -> float:
    try:
        from pystoi import stoi
        return float(stoi(ref, est, sr, extended=False))
    except Exception:
        return float("nan")


def evaluate(ref: np.ndarray, est: np.ndarray, sr: int = 16000) -> dict:
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    return {
        "wb_pesq": compute_pesq(ref, est, sr, "wb"),
        "nb_pesq": compute_pesq(ref, est, sr, "nb"),
        "stoi": compute_stoi(ref, est, sr),
        "si_sdr": si_sdr(est, ref),
    }
