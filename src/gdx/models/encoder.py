"""The layout-aware encoder: embeddings plus stacked attention blocks.

No pretrained checkpoint, no ``transformers``, no ``detectron2``. That is a
deliberate constraint, not a shortcut: the claim under test is about the *output
interface* (selection versus generation), and the cleanest way to isolate it is
to hold one small, fully-visible backbone fixed and change only the head. A
pretrained encoder would make the comparison depend on what the checkpoint had
already memorised about invoices.

Input signals, all summed at the bottom:

* word identity (closed lexicon + hash buckets),
* surface shape class,
* numeric magnitude bucket,
* a linear map of raw geometry,
* reading-order sinusoids (always),
* 2-D layout Fourier features (``use_2d_pos``).
"""

from __future__ import annotations

import torch
from torch import nn

from gdx.config import ModelConfig
from gdx.data.dataset import DocumentBatch
from gdx.data.featurise import N_GEOM, N_MAG_BUCKETS, N_SHAPE_CLASSES
from gdx.models.attention import EncoderLayer
from gdx.models.position import Box2DEncoding, Sinusoidal1D, SpatialBias


class TokenEmbedding(nn.Module):
    """Sum of identity, shape, magnitude and geometry embeddings."""

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.word = nn.Embedding(vocab_size, d, padding_idx=0)
        self.shape = nn.Embedding(N_SHAPE_CLASSES, d)
        self.mag = nn.Embedding(N_MAG_BUCKETS, d)
        self.geom = nn.Linear(N_GEOM, d)
        self.order = Sinusoidal1D(d)
        self.box = Box2DEncoding(cfg.n_pos_bands, d) if cfg.use_2d_pos else None
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)
        # Scale word embeddings like a transformer input so the summed signals
        # start at comparable magnitudes; without it the identity term dominates
        # and the geometry terms take most of training to become useful.
        self.scale = float(d) ** 0.5

    def forward(self, batch: DocumentBatch) -> torch.Tensor:
        """``DocumentBatch -> (B, L, d_model)``."""
        x = self.word(batch.word_ids) * self.scale
        x = x + self.shape(batch.shape_ids) + self.mag(batch.mag_ids)
        x = x + self.geom(batch.geom)
        x = x + self.order(batch.word_ids.shape[1]).to(x.dtype)
        if self.box is not None:
            x = x + self.box(batch.geom)
        return self.drop(self.norm(x))


class LayoutEncoder(nn.Module):
    """Token embedding plus ``n_layers`` pre-norm attention blocks.

    Attributes:
        spatial: The relative-geometry attention bias, or ``None`` when
            ``use_spatial_bias`` is off. It is computed once per forward pass and
            shared across layers, which is both cheaper and the standard T5
            arrangement -- a per-layer table would triple the ablation's degrees
            of freedom for no stated reason.
    """

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg, vocab_size)
        self.spatial = (
            SpatialBias(cfg.n_heads, cfg.n_spatial_buckets) if cfg.use_spatial_bias else None
        )
        self.layers = nn.ModuleList(
            EncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, batch: DocumentBatch) -> torch.Tensor:
        """``DocumentBatch -> (B, L, d_model)`` contextual token states."""
        x = self.embed(batch)
        bias = None
        if self.spatial is not None:
            bias = self.spatial(batch.geom, batch.page_ids)
        for layer in self.layers:
            x = layer(x, mask=batch.mask, bias=bias)
        return self.norm(x)

    def pool(self, states: torch.Tensor) -> torch.Tensor:
        """Document summary: the ``[CLS]`` state at index 0.

        Used only by the generative head, which needs a single conditioning
        vector. The span head never pools -- it reads token states directly,
        which is exactly why its output is grounded.
        """
        return states[:, 0]
