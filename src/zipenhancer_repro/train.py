"""Paper-consistent ZipEnhancer training for VoiceBank+DEMAND.

Stack:
    backbone  : vendored official-compatible ZipEnhancer architecture
    losses    : MP-SENet multi-loss + PESQ-GAN discriminator
    optimizer : icefall ScaledAdam + Eden
    data      : VoiceBank+DEMAND 16 kHz

Usage:
    python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml --smoke
    python -m zipenhancer_repro.train --config configs/zipenhancer_s.yaml
"""
from __future__ import annotations

import argparse
import os
import time

import librosa
import matplotlib
matplotlib.use("Agg")  # headless: training is a background process, no display
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.vb_dataset import VoiceBankDataset
from .losses.mpsenet_loss import MetricDiscriminator, batch_pesq, generator_loss
from .models.backbone import build_backbone, mag_pha_stft, mag_pha_istft
from .optim.icefall_optim import build_optimizer, set_batch_count


def _spec_figure(samp, n_fft, hop, win):
    """Stacked log-magnitude spectrograms (noisy / clean / enhanced) for one sample.

    Same STFT params as training (n_fft=400/hop=100/win=400) so the picture matches
    what the model actually sees. Returns a matplotlib Figure (caller must close it).
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, key in zip(axes, ("noisy", "clean", "enhanced")):
        S = np.abs(librosa.stft(samp[key], n_fft=n_fft, hop_length=hop, win_length=win))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        im = ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma",
                       vmin=-80, vmax=0)
        ax.set_title(key)
        ax.set_ylabel("freq bin")
        fig.colorbar(im, ax=ax, format="%+2.0f dB")
    axes[-1].set_xlabel("frame")
    pesq_str = f"{samp['pesq']:.3f}" if samp["pesq"] == samp["pesq"] else "nan"
    fig.suptitle(f"{samp['name']} | PESQ={pesq_str}")
    fig.tight_layout()
    return fig


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _generator_state(ck):
    return ck["generator"] if isinstance(ck, dict) and "generator" in ck else ck


def train(cfg, smoke=False, init_weights=None, resume=None, tb_run=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["train"]["seed"])
    if os.environ.get("DETECT_ANOMALY"):
        torch.autograd.set_detect_anomaly(True)
        print("[diag] torch.autograd anomaly detection ENABLED (slow)")
    s = cfg["stft"]
    n_fft, hop, win, cf = s["n_fft"], s["hop_length"], s["win_length"], s["compress_factor"]
    center = s.get("center", True)

    generator = build_backbone().to(device)
    discriminator = MetricDiscriminator().to(device)

    if init_weights:
        ck = torch.load(init_weights, map_location="cpu", weights_only=False)
        generator.load_state_dict(_generator_state(ck), strict=True)
        print(f"[init] loaded generator weights from {init_weights} (strict)")

    n_params = sum(p.numel() for p in generator.parameters()) / 1e6
    optim_g, sched_g, opt_name = build_optimizer(generator, cfg)
    # NOTE: base_lr (0.04) is for ScaledAdam (scale-invariant). The discriminator
    # uses vanilla AdamW, which needs a normal lr (~5e-4, as in MP-SENet). Reusing
    # 0.04 here makes the discriminator diverge and poison the generator -> NaN.
    disc_lr = cfg["optim"].get("disc_lr", 5e-4)
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.999))
    print(f"[init] gen params={n_params:.2f}M  stft_center={center}  "
          f"optimizer={opt_name}  disc_lr={disc_lr}  device={device}")

    if smoke:
        _smoke(cfg, generator, discriminator, optim_g, sched_g, optim_d, device,
               n_fft, hop, win, cf, center)
        return

    d = cfg["data"]["voicebank"]
    seg = int(cfg["train"]["segment_seconds"] * s["sample_rate"])
    ds = VoiceBankDataset(d["train_noisy"], d["train_clean"], s["sample_rate"], seg, split=True)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    print(f"[data] VoiceBank train pairs={len(ds)}  batches/epoch={len(loader)}")

    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    max_steps = cfg["train"]["max_steps"]
    log_every = cfg["train"]["log_every_steps"]
    save_every = cfg["train"]["save_every_steps"]
    eval_every = cfg["train"]["eval_every_steps"]
    bs = cfg["train"]["batch_size"]
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    # Eval config: subset (None=full 824), device (separate GPU recommended).
    eval_cfg = cfg.get("eval", {}) or {}
    eval_subset = eval_cfg.get("subset", None)
    eval_device_cfg = eval_cfg.get("device", "same")
    if eval_device_cfg == "same":
        eval_device = device
    elif eval_device_cfg == "cpu":
        eval_device = "cpu"
    else:
        eval_device = eval_device_cfg  # e.g. "cuda:1"
    print(f"[train] effective_bs={bs*grad_accum} (bs={bs} x grad_accum={grad_accum})  "
          f"eval_every={eval_every} eval_subset={eval_subset or 'full'} eval_device={eval_device}")

    # ---- loss weights from config (previously the config `loss:` block was IGNORED;
    # generator_loss silently used its hard-coded LOSS_WEIGHTS). Map config lambda_*
    # -> generator_loss keys, fall back to paper defaults. ----
    lc = cfg.get("loss", {})
    loss_weights = dict(
        mag=lc.get("lambda_mag", 0.9),
        pha=lc.get("lambda_pha", 0.3),
        com=lc.get("lambda_com", 0.1),
        stft=lc.get("lambda_stft", 0.1),
        metric=lc.get("lambda_pesq", 0.05),
        time=lc.get("lambda_time", 0.2),
    )
    use_pesq_gan = lc.get("use_pesq_gan", True)
    if not use_pesq_gan:
        loss_weights["metric"] = 0.0   # drop the PESQ-GAN generator term entirely
    print(f"[loss] weights={loss_weights}  use_pesq_gan={use_pesq_gan}")

    from torch.utils.tensorboard import SummaryWriter
    from .evaluate import evaluate_generator
    best_pesq = 0.0
    step, epoch = 0, 0
    # cumulative PESQ-GAN diagnostics: count how often a batch's PESQ computation
    # failed and the discriminator update was skipped. This used to be a SILENT
    # soft-skip; now it is counted + logged so a high rate surfaces instead of hiding.
    pesq_fail_total, pesq_batches_total = 0, 0
    # ---- resume BEFORE creating the SummaryWriter so we can reuse the original
    # run_name -> TensorBoard curve stays continuous across restarts. ----
    ck = None
    if resume:
        ckpt_path = resume
        if resume == "auto":
            latest = os.path.join(out_dir, "latest.pt")
            if not os.path.exists(latest):
                raise FileNotFoundError(f"--resume auto but no checkpoint at {latest}")
            ckpt_path = latest
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        generator.load_state_dict(ck["generator"], strict=True)
        discriminator.load_state_dict(ck["discriminator"], strict=True)
        optim_g.load_state_dict(ck["optim_g"])
        optim_d.load_state_dict(ck["optim_d"])
        step = ck["step"]; epoch = ck["epoch"]; best_pesq = ck.get("best_pesq", 0.0)
        pesq_fail_total = ck.get("pesq_fail_total", 0)
        pesq_batches_total = ck.get("pesq_batches_total", 0)
        print(f"[resume] loaded {ckpt_path} -> step={step} epoch={epoch} "
              f"best_pesq={best_pesq:.4f}")

    # ---- each training run gets its OWN tb subdirectory (run_name), so multiple
    # runs never mash into one curve. resume reuses the saved run_name. ----
    if ck and ck.get("run_name"):
        run_name = ck["run_name"]
    elif tb_run:
        run_name = tb_run
    else:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    tb_dir = os.path.join(out_dir, "tb", run_name)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    print(f"[tb] tensorboard logdir = {tb_dir}  (run={run_name})")

    def save_ckpt(path, tag):
        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "step": step, "epoch": epoch, "best_pesq": best_pesq,
            "run_name": run_name,
            "pesq_fail_total": pesq_fail_total,
            "pesq_batches_total": pesq_batches_total,
        }, path)
        print(f"[ckpt] saved {tag} (step={step})", flush=True)

    t0 = time.time()
    generator.train(); discriminator.train()
    micro_idx = 0  # current position inside a gradient-accumulation window
    while step < max_steps:
        for clean_audio, noisy_audio in loader:
            # drive the model's schedule-dependent regularizers (Balancer/Whiten/
            # ScheduledFloat) — REQUIRED for stable ScaledAdam training.
            set_batch_count(generator, step)

            clean_audio = clean_audio.to(device); noisy_audio = noisy_audio.to(device)
            one_labels = torch.ones(clean_audio.size(0), device=device)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, n_fft, hop, win, cf, center=center)
            noisy_mag, noisy_pha, _ = mag_pha_stft(noisy_audio, n_fft, hop, win, cf, center=center)

            out = generator(noisy_mag, noisy_pha)
            mag_g, pha_g, com_g = out[0], out[1], out[2]
            audio_g = mag_pha_istft(mag_g, pha_g, n_fft, hop, win, cf, center=center)
            mag_g_hat, _, com_g_hat = mag_pha_stft(audio_g, n_fft, hop, win, cf, center=center)

            # --- diagnostic: track magnitudes to localize divergence (fail-fast) ---
            diag = {
                "mag_g_max": mag_g.abs().max().item(),
                "audio_g_max": audio_g.abs().max().item(),
                "mag_g_hat_max": mag_g_hat.abs().max().item(),
                "out_finite": bool(torch.isfinite(mag_g).all() and torch.isfinite(pha_g).all()),
            }
            if not diag["out_finite"]:
                raise RuntimeError(
                    f"[DIVERGE] step {step}: generator output non-finite. "
                    f"mag_g_max={diag['mag_g_max']} audio_g_max={diag['audio_g_max']} "
                    f"(noisy_mag_max={noisy_mag.abs().max().item():.3f})")

            pesq_t = batch_pesq(list(clean_audio.detach().cpu().numpy()),
                                list(audio_g.detach().cpu().numpy()))

            # ---- discriminator (PESQ-GAN); skipped entirely when use_pesq_gan=False ----
            # With grad_accum>1: zero_grad ONLY at the first micro-batch of an
            # accumulation window; step ONLY at the last. Loss is divided by
            # grad_accum so the accumulated gradient matches a true bs*grad_accum batch.
            if use_pesq_gan:
                pesq_batches_total += 1
                if micro_idx == 0:
                    optim_d.zero_grad()
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g_hat.detach())
                loss_d_r = torch.nn.functional.mse_loss(one_labels, metric_r.flatten())
                if pesq_t is not None:
                    loss_d_g = torch.nn.functional.mse_loss(pesq_t.to(device), metric_g.flatten())
                    loss_d = loss_d_r + loss_d_g
                    (loss_d / grad_accum).backward()
                else:
                    # PESQ failed on >=1 sample -> no D regression target can be formed.
                    # We must skip the D step (cannot fabricate a target), but we COUNT it
                    # rather than hide it: a rising rate means generator output is off-range
                    # / the pesq lib is unhappy -> investigate, don't silently swallow.
                    pesq_fail_total += 1
                    loss_d = loss_d_r * 0
                metric_g2 = discriminator(clean_mag, mag_g_hat)
            else:
                loss_d = torch.zeros((), device=device)
                metric_g2 = one_labels  # placeholder; metric weight is 0 -> no contribution

            # ---- generator ----
            if micro_idx == 0:
                optim_g.zero_grad()
            loss_g, parts = generator_loss(
                clean_mag, mag_g, clean_pha, pha_g, clean_com, com_g, com_g_hat,
                clean_audio, audio_g, metric_g2, one_labels, weights=loss_weights)
            # No NaN/grad safety guards on purpose: training must be precise and
            # fail-fast. ScaledAdam's own clipping (clipping_scale=2.0) + the model's
            # Balancer/Whiten (driven by set_batch_count) are the legitimate stabilizers.
            (loss_g / grad_accum).backward()

            micro_idx += 1
            if micro_idx < grad_accum:
                # Still accumulating gradients — don't step the optimizer yet,
                # don't log/save/eval/increment `step` (step counts OPTIMIZER steps,
                # matching the paper's 600k-step schedule).
                continue
            micro_idx = 0
            if use_pesq_gan:
                optim_d.step()
            optim_g.step()
            if sched_g is not None:
                sched_g.step_batch(step)

            if step % log_every == 0:
                lr = optim_g.param_groups[0]["lr"]
                spd = (step + 1) / (time.time() - t0)
                # PESQ-GAN skip diagnostic: cumulative count of batches whose PESQ
                # computation failed -> D update skipped. Surfaced here (not just in
                # ckpt) so a rising rate is visible instead of silently swallowed.
                pf_rate = (pesq_fail_total / pesq_batches_total * 100.0) if pesq_batches_total else 0.0
                print(f"step {step:>7} ep{epoch} | G {loss_g.item():.3f} D {float(loss_d):.3f} "
                      f"| mag {parts['mag']:.3f} pha {parts['pha']:.3f} com {parts['com']:.3f} "
                      f"stft {parts['stft']:.3f} metric {parts['metric']:.3f} time {parts['time']:.3f} "
                      f"| |mag_g|max {diag['mag_g_max']:.2f} |aud|max {diag['audio_g_max']:.2f} "
                      f"| pesq_fail {pesq_fail_total}/{pesq_batches_total} ({pf_rate:.1f}%) "
                      f"| lr {lr:.2e} | {spd:.2f} it/s", flush=True)
                writer.add_scalar("train/G_loss", loss_g.item(), step)
                writer.add_scalar("train/D_loss", float(loss_d), step)
                for k, v in parts.items():
                    writer.add_scalar(f"loss/{k}", v, step)
                writer.add_scalar("diag/mag_g_max", diag["mag_g_max"], step)
                writer.add_scalar("diag/audio_g_max", diag["audio_g_max"], step)
                writer.add_scalar("diag/pesq_fail_rate", pf_rate, step)
                writer.add_scalar("diag/pesq_fail_total", pesq_fail_total, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/it_per_s", spd, step)

            # ---- checkpoint FIRST (before eval) so a crash/OOM in eval never costs
            # us the trained weights; latest.pt is always overwritten for --resume auto.
            if step > 0 and step % save_every == 0:
                save_ckpt(os.path.join(out_dir, "latest.pt"), "latest.pt")
                save_ckpt(os.path.join(out_dir, f"ckpt_{step:08d}.pt"),
                          f"ckpt_{step:08d}.pt")

            if step > 0 and step % eval_every == 0:
                # Eval uses CHUNKED overlap-add inference (zipenhancer_repro.infer.enhance_long,
                # called via evaluate.enhance_wav). Peak ~1GB regardless of utterance
                # length, so it runs comfortably on a separate GPU.
                # Default config: eval on cuda:1 (NOT the training GPU) -> zero
                # contention with training; best_pesq is now from the FULL 824-utt
                # test set (no subset bias).
                if eval_device != device and eval_device != "cpu":
                    # Move generator to the eval GPU, run, then move back.
                    generator.to(eval_device)
                    res = evaluate_generator(generator, cfg, eval_device,
                                             subset=eval_subset, verbose=True,
                                             tb_sample=True)
                    generator.to(device)
                elif eval_device == "cpu":
                    generator.cpu()
                    res = evaluate_generator(generator, cfg, "cpu",
                                             subset=eval_subset, verbose=True,
                                             tb_sample=True)
                    generator.to(device)
                else:
                    # eval on the same GPU as training (chunked => peak ~1GB, fits).
                    res = evaluate_generator(generator, cfg, device,
                                             subset=eval_subset, verbose=True,
                                             tb_sample=True)
                writer.add_scalar("eval/wb_pesq", res["wb_pesq"], step)
                writer.add_scalar("eval/stoi", res["stoi"], step)
                writer.add_scalar("eval/si_sdr", res["si_sdr"], step)
                # ---- listenable audio + spectrograms for TWO utterances per eval:
                #   fixed  -> same sentence every cycle (track one sentence's evolution)
                #   random -> rotates for coverage
                # Both reuse the waveform already enhanced inside evaluate_generator
                # (zero extra inference). Audio: peak-normalize the THREE clips by
                # their SHARED max so relative loudness stays comparable by ear.
                # Spectrograms use the training STFT params (n_fft=400/hop=100/win=400).
                for tag, samp in (res.get("samples") or {}).items():
                    sr_a = samp["sr"]
                    peak = max(float(np.abs(samp["noisy"]).max()),
                               float(np.abs(samp["clean"]).max()),
                               float(np.abs(samp["enhanced"]).max()), 1e-8)
                    for key in ("noisy", "clean", "enhanced"):
                        wav = np.clip(samp[key] / peak, -1.0, 1.0)
                        writer.add_audio(f"eval_audio_{tag}/{key}",
                                         torch.from_numpy(wav).float().unsqueeze(0),
                                         step, sample_rate=sr_a)
                    writer.add_text(f"eval_audio_{tag}/info",
                                    f"{samp['name']} | PESQ={samp['pesq']:.3f}", step)
                    fig = _spec_figure(samp, n_fft, hop, win)
                    writer.add_figure(f"eval_spec_{tag}", fig, step)
                    plt.close(fig)
                generator.train()
                if res["wb_pesq"] > best_pesq:
                    best_pesq = res["wb_pesq"]
                    save_ckpt(os.path.join(out_dir, "best.pt"),
                              f"best.pt (PESQ={best_pesq:.4f})")
                torch.cuda.empty_cache()

            step += 1
            if step >= max_steps:
                break
        epoch += 1
        if sched_g is not None:
            sched_g.step_epoch(epoch)

    save_ckpt(os.path.join(out_dir, "latest.pt"), "latest.pt")
    save_ckpt(os.path.join(out_dir, "final.pt"), "final.pt")
    writer.close()
    print("[done] training finished.")


def _smoke(cfg, generator, discriminator, optim_g, sched_g, optim_d, device,
           n_fft, hop, win, cf, center=True):
    print("[smoke] 5 steps on random 2s audio ...")
    L = int(cfg["train"]["segment_seconds"] * cfg["stft"]["sample_rate"])
    bs = cfg["train"]["batch_size"]
    generator.train(); discriminator.train()
    for step in range(5):
        clean_audio = torch.randn(bs, L, device=device) * 0.05
        noisy_audio = clean_audio + torch.randn(bs, L, device=device) * 0.05
        one = torch.ones(bs, device=device)
        clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, n_fft, hop, win, cf, center=center)
        noisy_mag, noisy_pha, _ = mag_pha_stft(noisy_audio, n_fft, hop, win, cf, center=center)
        out = generator(noisy_mag, noisy_pha)
        mag_g, pha_g, com_g = out[0], out[1], out[2]
        audio_g = mag_pha_istft(mag_g, pha_g, n_fft, hop, win, cf, center=center)
        mag_g_hat, _, com_g_hat = mag_pha_stft(audio_g, n_fft, hop, win, cf, center=center)

        optim_d.zero_grad()
        metric_r = discriminator(clean_mag, clean_mag)
        metric_g = discriminator(clean_mag, mag_g_hat.detach())
        loss_d = torch.nn.functional.mse_loss(one, metric_r.flatten()) + \
            torch.nn.functional.mse_loss(one * 0.5, metric_g.flatten())
        loss_d.backward(); optim_d.step()

        optim_g.zero_grad()
        metric_g2 = discriminator(clean_mag, mag_g_hat)
        loss_g, parts = generator_loss(clean_mag, mag_g, clean_pha, pha_g, clean_com,
                                       com_g, com_g_hat, clean_audio, audio_g, metric_g2, one)
        loss_g.backward(); optim_g.step()
        if sched_g is not None:
            sched_g.step_batch(step)
        lr = optim_g.param_groups[0]["lr"]
        print(f"[smoke] step {step} | G {loss_g.item():.3f} D {loss_d.item():.3f} "
              f"| in {tuple(noisy_audio.shape)} out {tuple(audio_g.shape)} | lr {lr:.2e}")
    print("[smoke] OK — backbone+losses+ScaledAdam/Eden+discriminator all wired.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/zipenhancer_s.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init-weights", default=None,
                    help="optional generator checkpoint or official pytorch_model.bin")
    ap.add_argument("--optim", default=None, choices=["scaled_adam", "adamw"],
                    help="override optimizer (scaled_adam=paper main, adamw=paper ablation/stable)")
    ap.add_argument("--resume", default=None,
                    help="resume from a full-state ckpt path, or 'auto' for <out_dir>/latest.pt")
    ap.add_argument("--run-name", default=None,
                    help="tensorboard run name (subdir under tb/); default = timestamp run_YYYYmmdd_HHMMSS")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.optim:
        cfg["optim"]["name"] = args.optim
    train(cfg, smoke=args.smoke, init_weights=args.init_weights,
          resume=args.resume, tb_run=args.run_name)


if __name__ == "__main__":
    main()
