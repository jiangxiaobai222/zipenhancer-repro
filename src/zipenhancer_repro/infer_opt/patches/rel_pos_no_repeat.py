"""Patch (B): kill the .repeat(batch*heads) in rel-pos gather.

The original implementation in
  vendor/zipenhancer_community/models/layers/zipformer.py
  RelPositionMultiheadAttentionWeights.forward  (around lines 676-685)
materializes an int64 `indexes` tensor of shape (H*B*T, T) by repeating
a (T, T) index pattern H*B times. Since dual-path Zipformer folds the
freq dim (~101) into the attention batch dim, this becomes the dominant
memory hog for full-utterance inference.

This patch replaces the index construction + reshape + gather with a
mathematically equivalent operation that uses `torch.expand` (zero-copy
stride view) for the index tensor, so the int64 indices stay at (T, T)
size regardless of H, B, freq.

Correctness: verified by `zipenhancer_repro.infer_opt.verify_numeric` (max-abs-diff vs
original ≤ 1e-5 in fp32). The model architecture & state_dict are not
changed; `verify_backbone.py` still passes strict-load.

Usage:
    from zipenhancer_repro.infer_opt.patches import rel_pos_no_repeat
    rel_pos_no_repeat.apply()
    # ... run inference ...
    rel_pos_no_repeat.revert()   # optional
"""
from __future__ import annotations

import random
from typing import Optional

import torch
from torch import Tensor, nn

from ...vendor.zipenhancer_community.models.layers.zipformer import RelPositionMultiheadAttentionWeights
from ...vendor.zipenhancer_community.models.layers.scaling import (
    softmax, penalize_abs_values_gt,
)

# Save the original forward so we can revert.
_ORIG_FORWARD = RelPositionMultiheadAttentionWeights.forward
_PATCHED = False


def _patched_forward(
    self,
    x: Tensor,
    pos_emb: Tensor,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
) -> Tensor:
    """Drop-in replacement for RelPositionMultiheadAttentionWeights.forward.

    Identical to the original except for the rel→abs gather block, which uses
    `abs_idx.expand(...)` (zero-copy view) instead of `rows.repeat(B*H)` (a
    full int64 materialization).
    """
    x = self.in_proj(x)
    query_head_dim = self.query_head_dim
    pos_head_dim = self.pos_head_dim
    num_heads = self.num_heads

    seq_len, batch_size, _ = x.shape
    query_dim = query_head_dim * num_heads

    # self-attention
    q = x[..., 0:query_dim]
    k = x[..., query_dim:2 * query_dim]
    p = x[..., 2 * query_dim:]
    assert p.shape[-1] == num_heads * pos_head_dim

    q = self.copy_query(q)
    k = self.whiten_keys(self.balance_keys(k))
    p = self.copy_pos_query(p)

    q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
    p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
    k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

    q = q.permute(2, 1, 0, 3)  # (H, B, T1, D_q)
    p = p.permute(2, 1, 0, 3)  # (H, B, T1, D_p)
    k = k.permute(2, 1, 3, 0)  # (H, B, D_k, T2)

    attn_scores = torch.matmul(q, k)

    use_pos_scores = False
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        use_pos_scores = True
    elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
        use_pos_scores = True

    if use_pos_scores:
        # Robust fp16 compat: pos_emb is buffered as the dtype of the very first
        # input that called extend_pe(); if model was later .half()'d, pos_emb
        # can still be fp32 while linear_pos.weight is fp16. Cast to match.
        pos_emb = pos_emb.to(self.linear_pos.weight.dtype)
        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads,
                                  pos_head_dim).permute(2, 0, 3, 1)
        # (H, {1 or B}, D_p, 2T-1)

        # pos_scores: (H, B, T1, 2T-1)
        pos_scores = torch.matmul(p, pos_emb)
        (H, B, T1, n) = pos_scores.shape
        # T1 == seq_len here (no chunking in the offline path).
        assert T1 == seq_len, (T1, seq_len)

        # === BEGIN PATCH (B): zero-copy expanded gather index ===
        # Equivalent to original (verified numerically):
        #   rows = torch.arange(T1-1, -1, -1)              # (T,)
        #   cols = torch.arange(T)                          # (T,)
        #   rows.repeat(B*H).unsqueeze(-1) + cols           # (B*H*T, T) ← waste
        # New: abs_idx is just (T, T) and broadcast via expand (no realloc).
        rows = torch.arange(seq_len - 1, -1, -1, device=pos_scores.device)
        cols = torch.arange(seq_len, device=pos_scores.device)
        abs_idx = rows.unsqueeze(-1) + cols.unsqueeze(0)    # (T, T) int64
        # expand to (H, B, T, T) is a stride view, NOT a copy.
        idx_view = abs_idx.expand(H, B, T1, seq_len)
        # gather along last dim. pos_scores stays (H, B, T, 2T-1) (no reshape).
        pos_scores = pos_scores.gather(dim=-1, index=idx_view)
        # === END PATCH (B) ===

        if self.training:
            attn_scores = attn_scores + pos_scores
        else:
            attn_scores = attn_scores + pos_scores

    if torch.jit.is_scripting() or torch.jit.is_tracing():
        pass
    elif self.training and random.random() < 0.1:
        attn_scores = penalize_abs_values_gt(
            attn_scores, limit=25.0, penalty=1.0e-04, name=self.name)

    assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

    if attn_mask is not None:
        assert attn_mask.dtype == torch.bool
        attn_scores = attn_scores.masked_fill(attn_mask, -1000)

    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, seq_len)
        attn_scores = attn_scores.masked_fill(
            key_padding_mask.unsqueeze(1), -1000)

    attn_weights = softmax(attn_scores, dim=-1)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training)
    return attn_weights


def apply() -> None:
    """Activate the patch (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    RelPositionMultiheadAttentionWeights.forward = _patched_forward
    _PATCHED = True


def revert() -> None:
    """Roll back to the original implementation."""
    global _PATCHED
    if not _PATCHED:
        return
    RelPositionMultiheadAttentionWeights.forward = _ORIG_FORWARD
    _PATCHED = False


def is_active() -> bool:
    return _PATCHED


__all__ = ["apply", "revert", "is_active"]
