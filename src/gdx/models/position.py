"""Hand-rolled positional encodings: reading order, 2-D layout, spatial bias.

Three separate signals, kept separate on purpose so the ablation means something.

**Reading order** (:class:`Sinusoidal1D`) is always on. If ``use_2d_pos=False``
removed *all* position information the ablation would be comparing "layout-aware"
against "bag of tokens", and any gap would be uninterpretable. The honest
contrast is **2-D layout position versus 1-D reading-order position**, which is
what this module implements.

**2-D layout position** (:class:`Box2DEncoding`) expands five geometric scalars
-- box centre, box size, page fraction -- into a Fourier feature bank and
projects it to the model width. The bank is the standard trick from Tancik et al.
(2020) / NeRF: a linear layer on raw coordinates cannot represent "these two
tokens are on the same row" without an enormous Lipschitz constant, whereas
``sin(2^k pi x)`` makes near-equality of ``y`` a low-frequency agreement.

**Spatial attention bias** (:class:`SpatialBias`) is a *relative* signal, and
that is the part absolute encodings cannot supply. "The value is to the right of
its label" and "the value is on the row below its label" are relations, not
positions. The bias is a learned table indexed by signed log-scale buckets of
``(dx, dy)`` plus a same-page flag, added to the attention logits before the
softmax -- the T5 relative-position mechanism (Raffel et al., 2020) generalised
from one dimension to two.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class Sinusoidal1D(nn.Module):
    """Fixed sinusoidal encoding of reading-order index.

    Not learned, so it extrapolates to sequences longer than anything seen in
    training -- which matters because multi-page invoices are longer than the
    single-page ones that dominate the training set.
    """

    def __init__(self, d_model: int, max_len: int = 1024) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for a sinusoidal encoding, got {d_model}")
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(pos * div)
        table[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("table", table, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:
        """``(1, seq_len, d_model)``.

        Raises:
            ValueError: If ``seq_len`` exceeds the precomputed table, instead of
                silently wrapping or truncating.
        """
        if seq_len > self.table.shape[0]:
            raise ValueError(f"sequence of {seq_len} exceeds max_len {self.table.shape[0]}")
        return self.table[:seq_len].unsqueeze(0)


class Box2DEncoding(nn.Module):
    """Fourier features over box geometry, projected to the model width.

    Args:
        n_bands: Frequency bands per input scalar. Frequencies are ``2^k`` for
            ``k`` in ``[0, n_bands)``, so the highest band resolves features of
            size ``2^-(n_bands-1)`` of the page: at 8 bands, about 0.8% of the
            page height, which is finer than one text line.
        d_model: Output width.
        n_inputs: Geometric scalars consumed from the front of the geometry
            vector. The featuriser orders them ``x0, y0, x1, y1, cx, cy, w, h,
            page_frac, is_last_page``; this module uses the last six, which are
            the scale-invariant ones.
    """

    #: Indices into the featuriser's geometry vector: cx, cy, w, h, page_frac.
    GEOM_SLICE = (4, 5, 6, 7, 8)

    def __init__(self, n_bands: int, d_model: int) -> None:
        super().__init__()
        if n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {n_bands}")
        self.n_bands = int(n_bands)
        self.n_inputs = len(self.GEOM_SLICE)
        freqs = 2.0 ** torch.arange(n_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        self.proj = nn.Linear(self.n_inputs * 2 * n_bands, d_model)

    @property
    def n_features(self) -> int:
        return self.n_inputs * 2 * self.n_bands

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """``(B, L, N_GEOM) -> (B, L, d_model)``."""
        picked = geom[..., list(self.GEOM_SLICE)]  # (B, L, n_inputs)
        scaled = picked.unsqueeze(-1) * self.freqs  # (B, L, n_inputs, n_bands)
        feats = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return self.proj(feats.flatten(start_dim=-2))


def bucket_signed_log(delta: torch.Tensor, n_buckets: int, max_abs: float = 1.0) -> torch.Tensor:
    """Signed log-scale bucket index for a relative offset.

    Buckets are symmetric about zero with ``n_buckets`` total, so the middle
    bucket is "same position" and resolution is finest near zero -- which is
    where layout relations live. A label and its value are a few percent of the
    page apart; a label and an unrelated token are tens of percent apart, and the
    difference between 30% and 60% carries almost no information, so a log scale
    is the right allocation of table capacity.

    Args:
        delta: Any shape of offsets.
        n_buckets: Odd number of buckets. Even values are made odd by
            subtracting one so that a zero bucket exists.
        max_abs: Offset magnitude mapped to the outermost bucket.

    Returns:
        Long tensor of indices in ``[0, n_buckets)``.
    """
    n = int(n_buckets)
    if n < 3:
        raise ValueError(f"n_buckets must be >= 3, got {n_buckets}")
    if n % 2 == 0:
        n -= 1
    half = (n - 1) // 2
    mag = (delta.abs() / max(1e-6, max_abs)).clamp(0.0, 1.0)
    # log1p(mag * (e - 1)) maps [0, 1] -> [0, 1] monotonically with a dense
    # low end; scaling by `half` and rounding gives the bucket offset.
    level = torch.log1p(mag * (math.e - 1.0)).clamp(0.0, 1.0)
    offset = torch.round(level * half).to(torch.long)
    signed = torch.where(delta >= 0, offset, -offset)
    return (signed + half).clamp(0, n - 1)


class SpatialBias(nn.Module):
    """Learned per-head attention bias from relative 2-D geometry.

    The table is ``(n_heads, n_buckets, n_buckets)`` over bucketed ``(dx, dy)``
    plus a ``(n_heads, 2)`` same-page term. Parameter cost at the shipped setting
    is ``4 * 9 * 9 + 4 * 2 = 332`` -- negligible, which is the point: the
    mechanism buys a relational prior almost for free, and the ablation measures
    whether the prior is worth anything.
    """

    def __init__(self, n_heads: int, n_buckets: int = 9) -> None:
        super().__init__()
        n = int(n_buckets) - (1 - int(n_buckets) % 2)
        self.n_buckets = n
        self.n_heads = int(n_heads)
        self.table = nn.Parameter(torch.zeros(n_heads, n, n))
        self.page_bias = nn.Parameter(torch.zeros(n_heads, 2))
        nn.init.normal_(self.table, std=0.02)

    def forward(self, geom: torch.Tensor, page_ids: torch.Tensor) -> torch.Tensor:
        """``(B, L, N_GEOM), (B, L) -> (B, n_heads, L, L)`` additive bias."""
        cx = geom[..., 4]
        cy = geom[..., 5]
        dx = cx.unsqueeze(1) - cx.unsqueeze(2)  # (B, L_q, L_k) as key minus query
        dy = cy.unsqueeze(1) - cy.unsqueeze(2)
        bx = bucket_signed_log(dx.transpose(1, 2), self.n_buckets)
        by = bucket_signed_log(dy.transpose(1, 2), self.n_buckets)
        bias = self.table[:, bx, by]  # (n_heads, B, L, L)
        same = (page_ids.unsqueeze(2) == page_ids.unsqueeze(1)).long()  # (B, L, L)
        bias = bias + self.page_bias[:, same.reshape(-1)].reshape(
            self.n_heads, *same.shape
        )
        return bias.permute(1, 0, 2, 3)
