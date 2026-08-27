"""Hand-rolled multi-head self-attention with an optional 2-D spatial bias.

Written out rather than delegated to ``nn.MultiheadAttention`` for two reasons.
The first is that the spatial bias has to be added to the attention *logits*, and
the fused path does not expose them in a form this module can extend portably.
The second is that the whole point of the repository is a readable, self-contained
implementation: the mechanism that makes the encoder layout-aware is fifteen lines
here and can be checked against a hand-computed reference in the tests, which is
not true of a fused kernel.

Masking convention: ``mask`` is ``True`` where a position is *valid*. Invalid
keys receive ``-inf`` before the softmax. A row that is entirely masked would
produce ``NaN`` from ``softmax(-inf)``; that cannot happen here because index 0
is the ``[CLS]`` slot and is always valid, but :meth:`Attention.forward` still
guards it, because a silent ``NaN`` propagating into a loss is expensive to find.
"""

from __future__ import annotations

import math

import torch
from torch import nn

#: Additive mask sentinel. A literal ``-inf`` in the logits makes the softmax
#: gradient NaN for a row that is entirely masked; a large finite value keeps the
#: post-softmax weight below 1e-30 while staying differentiable.
NEG_INF = -1e9


class Attention(nn.Module):
    """Multi-head self-attention.

    Args:
        d_model: Model width. Must be divisible by ``n_heads``.
        n_heads: Head count.
        dropout: Dropout on the attention weights and on the output projection.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_head = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``(B, L, d_model) -> (B, L, d_model)``.

        Args:
            x: Input sequence.
            mask: ``(B, L)`` bool, ``True`` where valid.
            bias: ``(B, n_heads, L, L)`` additive attention bias, or ``None``.
            return_weights: Also return the post-softmax weights, which the
                tests and the attention figure both need.
        """
        b, length, _ = x.shape
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if bias is not None:
            logits = logits + bias
        if mask is not None:
            keep = mask[:, None, None, :].expand(b, self.n_heads, length, length)
            logits = logits.masked_fill(~keep, NEG_INF)
            # A fully-masked row cannot happen (index 0 is always valid) but the
            # guard costs nothing and a NaN here would be silent.
            all_masked = (~mask).all(dim=-1)
            if bool(all_masked.any()):
                logits = logits.masked_fill(all_masked[:, None, None, None], 0.0)

        weights = torch.softmax(logits, dim=-1)
        weights = self.attn_drop(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(b, length, self.d_model)
        out = self.out_drop(self.out_proj(out))
        if return_weights:
            return out, weights
        return out

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, length, _ = t.shape
        return t.reshape(b, length, self.n_heads, self.d_head).transpose(1, 2)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block.

    Pre-norm (``x + attn(norm(x))``) rather than post-norm because this encoder
    trains for a few hundred steps with no warm-up budget to spare; post-norm
    needs a longer warm-up to avoid the early-training gradient spike, and
    spending steps on that would come out of the compute budget the whole project
    is constrained by.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask, bias=bias)
        return x + self.ff(self.norm2(x))
