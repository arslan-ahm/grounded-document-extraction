"""The assembled model: one encoder, one of two heads.

``model.head`` is the *only* configuration difference between the method and its
reference baseline. Everything else -- width, depth, positional encodings,
optimiser, schedule, data, seed -- is held fixed by construction rather than by
convention, because separate scripts per arm are how an incidental difference in
schedule gets mistaken for a difference in method.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gdx.config import ModelConfig
from gdx.data.dataset import DocumentBatch
from gdx.models.encoder import LayoutEncoder
from gdx.models.heads import GenerativeHead, GenPrediction, SpanHead, SpanPrediction

HEADS = ("span", "generative")


class GDXModel(nn.Module):
    """Layout encoder plus a span-selection or a generative head.

    Args:
        cfg: Model configuration. ``cfg.head`` picks the output interface.
        vocab_size: Rows in the word embedding table.

    Raises:
        ValueError: On an unknown head name, listing the valid ones. A typo here
            would otherwise silently train the wrong arm.
    """

    def __init__(self, cfg: ModelConfig, vocab_size: int) -> None:
        super().__init__()
        if cfg.head not in HEADS:
            raise ValueError(f"unknown model.head={cfg.head!r}; expected one of {HEADS}")
        self.cfg = cfg
        self.encoder = LayoutEncoder(cfg, vocab_size)
        self.span_head = SpanHead(cfg) if cfg.head == "span" else None
        self.gen_head = GenerativeHead(cfg) if cfg.head == "generative" else None

    @property
    def head_name(self) -> str:
        return self.cfg.head

    def encode(self, batch: DocumentBatch) -> torch.Tensor:
        return self.encoder(batch)

    def forward(self, batch: DocumentBatch) -> dict[str, torch.Tensor]:
        """Head-specific logits, keyed so the trainer does not branch on type."""
        states = self.encode(batch)
        if self.span_head is not None:
            start, end = self.span_head(states, batch.mask)
            return {"states": states, "start_logits": start, "end_logits": end}
        assert self.gen_head is not None
        return {
            "states": states,
            "char_logits": self.gen_head(states, batch.mask, batch.chars),
        }

    def loss(self, batch: DocumentBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """Scalar training loss and a dict of scalars for the history log."""
        out = self.forward(batch)
        if self.span_head is not None:
            loss = self.span_head.loss(
                out["start_logits"],
                out["end_logits"],
                batch.start,
                batch.end,
                label_smoothing=0.0,
            )
        else:
            assert self.gen_head is not None
            loss = self.gen_head.loss(out["char_logits"], batch.chars)
        return loss, {"loss": float(loss.detach())}

    @torch.no_grad()
    def predict(
        self, batch: DocumentBatch, top_k: int = 5
    ) -> list[list[SpanPrediction]] | list[list[GenPrediction]]:
        """Decode one batch into per-field head predictions."""
        was_training = self.training
        self.eval()
        try:
            states = self.encode(batch)
            if self.span_head is not None:
                start, end = self.span_head(states, batch.mask)
                lengths = [len(d.tokens) for d in batch.docs]
                # Page indices are passed so the candidate space cannot contain a
                # span spliced across a page break; see SpanHead._decode_one.
                pages = [
                    np.asarray([t.page for t in d.tokens], dtype=np.int64) for d in batch.docs
                ]
                return self.span_head.decode(start, end, lengths, top_k=top_k, pages=pages)
            assert self.gen_head is not None
            return self.gen_head.decode(states, batch.mask)
        finally:
            self.train(was_training)


def build_model(cfg: ModelConfig, vocab_size: int, seed: int | None = None) -> GDXModel:
    """Construct a model, optionally under a fixed initialisation seed.

    The seed is applied *here* rather than globally so that the two arms get
    identical encoder initialisations when run with the same seed -- the head
    parameters differ in shape and therefore consume the RNG differently, so the
    encoder is built first and the head second.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return GDXModel(cfg, vocab_size)
