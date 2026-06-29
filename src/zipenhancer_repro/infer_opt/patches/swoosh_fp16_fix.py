"""Patch: make SwooshL/RFunction dtype-aware for FP16 inference.

Bug in the vendored community scaling.py: SwooshL/RFunction.forward
upcasts fp16 input to fp32 internally for numerical stability, but the
later check `if x.dtype == torch.float16: y = y.to(torch.float16)` runs
AFTER `x` has been overwritten with the fp32 tensor — so the condition
is never satisfied and the function returns fp32 even for fp16 input.

This breaks .half() inference: the next layer (e.g. F.linear with fp16
weights) then errors with "mat1 Float, mat2 Half".

Fix: snapshot the original dtype before the upcast, use that for the
output cast decision. Mathematically equivalent for fp32 input (no-op
cast back to fp32).

Important: this patch ONLY affects inference. For training (where these
autograd Functions provide the int8-quantized derivative trick used by
icefall), the upstream behavior is preserved when input is fp32.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ...vendor.zipenhancer_community.models.layers.scaling import SwooshLFunction, SwooshRFunction

_ORIG_L_FWD = SwooshLFunction.forward
_ORIG_R_FWD = SwooshRFunction.forward
_PATCHED = False


@staticmethod
def _patched_swoosh_l_forward(ctx, x: Tensor) -> Tensor:
    """SwooshL forward that preserves fp16 dtype across the internal fp32 upcast."""
    requires_grad = x.requires_grad
    in_dtype = x.dtype                          # remember original dtype
    if in_dtype == torch.float16:
        x = x.to(torch.float32)

    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    coeff = -0.08

    with torch.amp.autocast('cuda', enabled=False):
        with torch.enable_grad():
            x = x.detach()
            x.requires_grad = True
            y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

            if not requires_grad:
                # inference path: just cast back if input was fp16
                if in_dtype == torch.float16:
                    y = y.to(torch.float16)
                return y

            y.backward(gradient=torch.ones_like(y))
            grad = x.grad
            floor = coeff
            ceil = 1.0 + coeff + 0.005
            _diff = (grad - floor) * (255.0 / (ceil - floor))
            d_scaled = _diff + torch.rand_like(grad)
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
            # cast output back to fp16 if input was fp16 OR amp is active
            if in_dtype == torch.float16 or torch.is_autocast_enabled():
                y = y.to(torch.float16)
            return y


@staticmethod
def _patched_swoosh_r_forward(ctx, x: Tensor) -> Tensor:
    """SwooshR forward that preserves fp16 dtype across the internal fp32 upcast."""
    requires_grad = x.requires_grad
    in_dtype = x.dtype
    if in_dtype == torch.float16:
        x = x.to(torch.float32)

    zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    with torch.amp.autocast('cuda', enabled=False):
        with torch.enable_grad():
            x = x.detach()
            x.requires_grad = True
            y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

            if not requires_grad:
                if in_dtype == torch.float16:
                    y = y.to(torch.float16)
                return y

            y.backward(gradient=torch.ones_like(y))
            grad = x.grad
            floor = -0.08
            ceil = 0.925
            d_scaled = ((grad - floor) * (255.0 / (ceil - floor))
                        + torch.rand_like(grad))
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
            if in_dtype == torch.float16 or torch.is_autocast_enabled():
                y = y.to(torch.float16)
            return y


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    SwooshLFunction.forward = _patched_swoosh_l_forward
    SwooshRFunction.forward = _patched_swoosh_r_forward
    _PATCHED = True


def revert() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    SwooshLFunction.forward = _ORIG_L_FWD
    SwooshRFunction.forward = _ORIG_R_FWD
    _PATCHED = False


def is_active() -> bool:
    return _PATCHED


__all__ = ["apply", "revert", "is_active"]
