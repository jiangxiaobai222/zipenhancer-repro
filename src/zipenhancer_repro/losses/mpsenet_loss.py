"""Paper-consistent losses & PESQ-GAN discriminator (ported verbatim from MP-SENet).

ZipEnhancer states its loss follows MP-SENet[16]/MP-SENet-Updated[34]:
    L = 0.9*L_mag + 0.3*L_pha + 0.1*L_com + 0.1*L_stft + 0.05*L_metric + 0.2*L_time
  - L_mag    : MSE on (compressed) magnitude
  - L_pha    : anti-wrapping phase loss = ip + gd + iaf
  - L_com    : MSE on complex spectrum * 2
  - L_stft   : consistency MSE(com_g, com_g_hat) * 2   (com_g_hat = STFT(iSTFT(out)))
  - L_metric : MSE(disc(clean_mag, mag_g_hat), 1)      (PESQ-GAN generator term)
  - L_time   : L1 on waveform

Discriminator target = normalized PESQ in [0,1] via (pesq-1)/3.5.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- phase loss -----------------------------
def anti_wrapping_function(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


def phase_losses(phase_r: torch.Tensor, phase_g: torch.Tensor):
    ip = torch.mean(anti_wrapping_function(phase_r - phase_g))
    gd = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=1) - torch.diff(phase_g, dim=1)))
    iaf = torch.mean(anti_wrapping_function(torch.diff(phase_r, dim=2) - torch.diff(phase_g, dim=2)))
    return ip, gd, iaf


# ----------------------------- PESQ helpers -----------------------------
def cal_pesq(clean: np.ndarray, est: np.ndarray, sr: int = 16000) -> float:
    try:
        from pesq import pesq
        return pesq(sr, clean, est, "wb")
    except Exception:
        return -1.0


def batch_pesq(clean_list, est_list):
    """Returns normalized PESQ tensor in [0,1], or None if any sample failed."""
    try:
        from joblib import Parallel, delayed
        scores = Parallel(n_jobs=15)(delayed(cal_pesq)(c, n) for c, n in zip(clean_list, est_list))
    except Exception:
        scores = [cal_pesq(c, n) for c, n in zip(clean_list, est_list)]
    scores = np.array(scores)
    if -1 in scores:
        return None
    scores = (scores - 1) / 3.5
    return torch.FloatTensor(scores)


# ----------------------------- discriminator -----------------------------
class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class MetricDiscriminator(nn.Module):
    def __init__(self, dim=16, in_channel=2):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.layers = nn.Sequential(
            sn(nn.Conv2d(in_channel, dim, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim, affine=True), nn.PReLU(dim),
            sn(nn.Conv2d(dim, dim * 2, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 2, affine=True), nn.PReLU(dim * 2),
            sn(nn.Conv2d(dim * 2, dim * 4, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 4, affine=True), nn.PReLU(dim * 4),
            sn(nn.Conv2d(dim * 4, dim * 8, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.InstanceNorm2d(dim * 8, affine=True), nn.PReLU(dim * 8),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(),
            sn(nn.Linear(dim * 8, dim * 4)), nn.Dropout(0.3), nn.PReLU(dim * 4),
            sn(nn.Linear(dim * 4, 1)), LearnableSigmoid1d(1),
        )

    def forward(self, x, y):
        xy = torch.stack((x, y), dim=1)  # [B,2,T,F]
        return self.layers(xy)


# ----------------------------- generator loss -----------------------------
# weights map to ZipEnhancer's lambda (mag/pha/com/stft/metric/time)
LOSS_WEIGHTS = dict(mag=0.9, pha=0.3, com=0.1, stft=0.1, metric=0.05, time=0.2)


def generator_loss(clean_mag, mag_g, clean_pha, pha_g, clean_com, com_g,
                   com_g_hat, clean_audio, audio_g, metric_g, one_labels,
                   weights=LOSS_WEIGHTS):
    loss_mag = F.mse_loss(clean_mag, mag_g)
    ip, gd, iaf = phase_losses(clean_pha, pha_g)
    loss_pha = ip + gd + iaf
    loss_com = F.mse_loss(clean_com, com_g) * 2
    loss_stft = F.mse_loss(com_g, com_g_hat) * 2
    loss_time = F.l1_loss(clean_audio, audio_g)
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
    total = (weights["mag"] * loss_mag + weights["pha"] * loss_pha
             + weights["com"] * loss_com + weights["stft"] * loss_stft
             + weights["metric"] * loss_metric + weights["time"] * loss_time)
    parts = dict(mag=loss_mag.item(), pha=loss_pha.item(), com=loss_com.item(),
                 stft=loss_stft.item(), metric=loss_metric.item(), time=loss_time.item())
    return total, parts
