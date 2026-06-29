"""Optimizer / scheduler factory + icefall training scaffolding (precise, no fallback).

The icefall Zipformer is trained with ScaledAdam in a way that the community AdamW
training omits. Replicating it is REQUIRED for stable/precise training:
  1) named parameter groups via get_parameter_groups_with_lrs (scalar params get
     scalar_lr_scale; lr_scale-tagged submodules get scaled lr)
  2) set_batch_count(model, step) every batch, which drives the model's internal
     Balancer / Whiten / ActivationBalancer / ScheduledFloat regularizers that keep
     activations & gradients in range (otherwise training diverges)
  3) Eden(..., warmup_start=0.1)

All borrowed verbatim from icefall (utils.py / train.py).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import torch.nn as nn

from ._icefall_optim import ScaledAdam, Eden


def get_parameter_groups_with_lrs(model: nn.Module, lr: float,
                                  include_names: bool = False,
                                  freeze_modules: List[str] = []) -> List[dict]:
    """Verbatim from icefall.utils — builds (named) parameter groups honoring
    per-module `lr_scale` attributes, for use with ScaledAdam."""
    flat_lr_scale = defaultdict(lambda: 1.0)
    for name, m in model.named_modules():
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale

    lr_to_params = defaultdict(list)
    for name, parameter in model.named_parameters():
        split_name = name.split(".")
        prefix = split_name[0]
        if prefix == "module":  # DDP
            module_name = split_name[1]
            if module_name in freeze_modules:
                continue
        elif prefix in freeze_modules:
            continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)

    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]


def set_batch_count(model: nn.Module, batch_count: float) -> None:
    """Drive the model's schedule-dependent regularizers (verbatim from icefall)."""
    if hasattr(model, "module"):  # DDP
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def build_optimizer(model, cfg):
    """Two paper-consistent recipes (selected by cfg.optim.name):
      - 'scaled_adam' : ZipEnhancer main recipe (ScaledAdam + Eden, lr=0.04)
      - 'adamw'       : ZipEnhancer ablation row "S(AdamW)" (AdamW lr=5e-4, known-stable;
                        paper reports only ~-0.04 PESQ). Used to isolate backbone/loss
                        correctness from ScaledAdam tuning.
    """
    import torch

    oc = cfg["optim"]
    name = oc.get("name", "scaled_adam").lower()

    if name == "adamw":
        lr = oc.get("adamw_lr", 5e-4)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                weight_decay=0.0)
        return opt, None, f"AdamW(lr={lr}) [ZipEnhancer S(AdamW) ablation]"

    eden = oc["eden"]
    base_lr = oc["base_lr"]
    opt = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=base_lr, include_names=True),
        lr=base_lr,
        clipping_scale=oc.get("clipping_scale", 2.0),
    )
    sched = Eden(opt, lr_batches=eden["step_size"], lr_epochs=eden["epoch_size"],
                 warmup_batches=eden["warmup_steps"], warmup_start=0.1)
    return opt, sched, "icefall:ScaledAdam+Eden (named-groups, warmup_start=0.1)"
