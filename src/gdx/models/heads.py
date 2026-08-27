"""The two output interfaces, over one shared encoder.

This module is where the repository's claim lives. Both heads read the same
encoder states, are trained by the same loop on the same data with the same
budget and seed, and differ in exactly one respect:

* :class:`SpanHead` emits **two indices**. Whatever string it produces is
  ``" ".join(doc.tokens[s:e+1])``, so a value absent from the document is not a
  low-probability output -- it is not in the output space at all.
* :class:`GenerativeHead` emits **characters**. It can produce any string over a
  70-character alphabet, which is the reference approach and is what makes a
  hallucinated total indistinguishable from a correctly-read one.

Both heads can abstain, and they must, or the comparison would be unfair: the
span head abstains by pointing both endpoints at the ``[CLS]`` null slot, and the
generative head by emitting the empty string. The generative head also attends
over the encoder states rather than over a pooled vector -- a pooled-only decoder
could not read a specific number off the page at all, and beating that would
prove nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as tf

from gdx.config import ModelConfig
from gdx.data.featurise import CHAR_BOS, CHAR_EOS, CHAR_PAD, N_CHARS, decode_chars
from gdx.data.schema import FIELDS
from gdx.models.attention import NEG_INF


@dataclass
class SpanPrediction:
    """One field's selected span, in *document* token coordinates.

    ``start == -1`` means the head abstained. ``prob`` is the probability of the
    chosen option under the distribution renormalised over the decision set
    (the null plus every admissible span), which is the quantity the calibration
    metrics consume -- an unnormalised span score is not a probability and
    calibrating it would be meaningless.
    """

    field_name: str
    start: int
    end: int
    prob: float
    null_prob: float
    candidates: list[tuple[int, int, float]] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.start < 0


class SpanHead(nn.Module):
    """Pointer head: per-field start and end distributions over tokens.

    The factorisation ``p(s, e) = p_start(s) p_end(e)`` is the standard
    extractive-QA one (Devlin et al., 2019). It is not exact -- start and end are
    not independent -- but the admissibility mask (``s <= e`` and
    ``e - s + 1 <= max_span_len``) plus renormalisation over the decision set
    recovers a proper distribution over the options the head can actually choose,
    which is all the downstream calibration needs.
    """

    def __init__(self, cfg: ModelConfig, n_fields: int = len(FIELDS)) -> None:
        super().__init__()
        self.n_fields = int(n_fields)
        self.max_span_len = int(cfg.max_span_len)
        self.proj = nn.Linear(cfg.d_model, 2 * self.n_fields)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, L, d), (B, L) -> two ``(B, F, L)`` logit tensors."""
        logits = self.proj(states)  # (B, L, 2F)
        start = logits[..., : self.n_fields].permute(0, 2, 1)
        end = logits[..., self.n_fields :].permute(0, 2, 1)
        keep = mask[:, None, :]
        start = start.masked_fill(~keep, NEG_INF)
        end = end.masked_fill(~keep, NEG_INF)
        return start, end

    def loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_target: torch.Tensor,
        end_target: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """Mean cross-entropy over both endpoints and all fields.

        Absent fields are *included*, with both targets at the null index, so
        abstention is learned by the same objective rather than by a separate
        classifier whose threshold would need its own tuning.
        """
        b, f, length = start_logits.shape
        flat_start = start_logits.reshape(b * f, length)
        flat_end = end_logits.reshape(b * f, length)
        loss_start = tf.cross_entropy(
            flat_start, start_target.reshape(-1), label_smoothing=label_smoothing
        )
        loss_end = tf.cross_entropy(
            flat_end, end_target.reshape(-1), label_smoothing=label_smoothing
        )
        return 0.5 * (loss_start + loss_end)

    @torch.no_grad()
    def decode(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        lengths: list[int],
        top_k: int = 5,
        pages: list[np.ndarray] | None = None,
    ) -> list[list[SpanPrediction]]:
        """Greedy decode with an admissibility mask, plus the top-``k`` runners-up.

        The runners-up are what the bounded verification loop re-selects from: a
        rejected candidate is masked out and the next admissible one is taken.

        Args:
            start_logits: ``(B, F, L)``.
            end_logits: ``(B, F, L)``.
            lengths: Real token count per document (excluding ``[CLS]``).
            top_k: Candidates retained per field.
            pages: Per document, the page index of each token. Required for the
                grounding guarantee -- see :func:`_decode_one`.

        Returns:
            ``B`` lists of ``F`` :class:`SpanPrediction`, in document coordinates.
        """
        p_start = torch.softmax(start_logits.float(), dim=-1).cpu().numpy()
        p_end = torch.softmax(end_logits.float(), dim=-1).cpu().numpy()
        out: list[list[SpanPrediction]] = []
        for b, n_tokens in enumerate(lengths):
            page = None if pages is None else pages[b]
            per_doc: list[SpanPrediction] = []
            for f, name in enumerate(FIELDS):
                per_doc.append(
                    _decode_one(
                        p_start[b, f],
                        p_end[b, f],
                        n_tokens,
                        self.max_span_len,
                        name,
                        top_k,
                        page,
                    )
                )
            out.append(per_doc)
        return out


def _decode_one(
    ps: np.ndarray,
    pe: np.ndarray,
    n_tokens: int,
    max_span_len: int,
    field_name: str,
    top_k: int,
    pages: np.ndarray | None = None,
) -> SpanPrediction:
    """Decode one field. Sequence index ``i`` is document token ``i - 1``.

    Vectorised over the ``(start, end)`` grid rather than looped. The loop
    version was 30x slower and would have made the *selection* head look slow in
    the latency benchmark against the generative head -- a measurement artefact
    that would have inverted the efficiency claim.

    **Page confinement is load-bearing, not a nicety.** A span whose endpoints
    sit on different pages concatenates the end of one page with the start of the
    next; its text is a sequence of document tokens but is *not* a contiguous
    region of the document, it has no union box, and the provenance predicate
    correctly refuses to find it. Allowing such spans in the candidate space broke
    the no-hallucination invariant -- ``test_untrained_span_model_never_hallucinates``
    caught it on 3 of 8 seeds with an emitted value of ``"of 2 Invoice No"``
    spliced across a page break. The guarantee holds only when the candidate space
    is exactly the set of spans the provenance check accepts, so the mask enforces
    it here. Pages are non-decreasing in reading order, so equality of the two
    endpoints implies every token between them is on the same page.
    """
    null = float(ps[0] * pe[0])
    if n_tokens <= 0:
        return SpanPrediction(field_name, -1, -1, 1.0, 1.0, [])
    outer = np.outer(ps[1 : n_tokens + 1], pe[1 : n_tokens + 1])
    rows = np.arange(n_tokens)[:, None]
    cols = np.arange(n_tokens)[None, :]
    admissible = (cols >= rows) & (cols - rows < max_span_len)
    if pages is not None:
        page = np.asarray(pages[:n_tokens])
        admissible = admissible & (page[:, None] == page[None, :])
    masked = np.where(admissible, outer, 0.0)
    total = null + float(masked.sum())
    if not np.isfinite(total) or total <= 0.0:
        return SpanPrediction(field_name, -1, -1, float("nan"), float("nan"), [])

    flat = masked.reshape(-1)
    k = int(min(top_k, int(admissible.sum())))
    if k <= 0:
        null_p = null / total
        return SpanPrediction(field_name, -1, -1, null_p, null_p, [])
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx], kind="stable")]
    cands = [
        (int(i // n_tokens), int(i % n_tokens), float(flat[i] / total))
        for i in idx
        if flat[i] > 0.0
    ]
    null_p = null / total
    if not cands or null_p >= cands[0][2]:
        return SpanPrediction(field_name, -1, -1, null_p, null_p, cands)
    s, e, p = cands[0]
    return SpanPrediction(field_name, s, e, p, null_p, cands)


@dataclass
class GenPrediction:
    """One field's generated string, with its sequence-level confidence.

    ``text`` is the decoded string; an empty string is the head's abstention.
    ``prob`` is the geometric mean of the per-character probabilities, which is
    length-normalised -- an arithmetic mean would make long values look
    systematically less confident than short ones and wreck the calibration
    comparison between the two heads.
    """

    field_name: str
    text: str
    prob: float

    @property
    def abstained(self) -> bool:
        return self.text == ""


class GenerativeHead(nn.Module):
    """Character-level attentional GRU decoder -- the reference approach.

    One decoder is shared across fields, conditioned on a field embedding, which
    keeps the parameter count comparable to the span head's single linear layer
    and avoids giving the baseline eight separate decoders' worth of capacity
    that the span head does not have.
    """

    def __init__(self, cfg: ModelConfig, n_fields: int = len(FIELDS)) -> None:
        super().__init__()
        self.n_fields = int(n_fields)
        self.max_len = int(cfg.dec_max_len)
        d, h = cfg.d_model, cfg.dec_hidden
        self.char_emb = nn.Embedding(N_CHARS, 32, padding_idx=CHAR_PAD)
        self.field_emb = nn.Embedding(self.n_fields, d)
        self.init_h = nn.Linear(2 * d, h)
        self.cell = nn.GRUCell(32 + d, h)
        # Additive attention over encoder states (Bahdanau et al., 2015). The
        # decoder must be able to read a specific number off the page; a
        # pooled-only conditioning would be a strawman baseline.
        self.attn_q = nn.Linear(h, d)
        self.attn_k = nn.Linear(d, d)
        self.attn_v = nn.Linear(d, 1)
        self.out = nn.Linear(h + d, N_CHARS)
        self.drop = nn.Dropout(cfg.dropout)

    def _attend(
        self, hidden: torch.Tensor, states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """``(B, h), (B, L, d), (B, L) -> (B, d)`` context vector."""
        scores = self.attn_v(
            torch.tanh(self.attn_k(states) + self.attn_q(hidden).unsqueeze(1))
        ).squeeze(-1)
        scores = scores.masked_fill(~mask, NEG_INF)
        return torch.bmm(torch.softmax(scores, dim=-1).unsqueeze(1), states).squeeze(1)

    def _step(
        self,
        char_ids: torch.Tensor,
        hidden: torch.Tensor,
        states: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._attend(hidden, states, mask)
        hidden = self.cell(torch.cat([self.char_emb(char_ids), context], dim=-1), hidden)
        logits = self.out(self.drop(torch.cat([hidden, context], dim=-1)))
        return logits, hidden

    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced logits: ``(B, L, d), (B, L), (B, F, T) -> (B, F, T-1, C)``."""
        b, _, d = states.shape
        pooled = states[:, 0]
        outs = []
        for f in range(self.n_fields):
            fid = torch.full((b,), f, dtype=torch.long, device=states.device)
            femb = self.field_emb(fid)
            hidden = torch.tanh(self.init_h(torch.cat([pooled, femb], dim=-1)))
            step_logits = []
            for t in range(targets.shape[2] - 1):
                logits, hidden = self._step(targets[:, f, t], hidden, states, mask)
                step_logits.append(logits)
            outs.append(torch.stack(step_logits, dim=1))
        del d
        return torch.stack(outs, dim=1)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over predicted characters, ignoring padding.

        ``ignore_index=CHAR_PAD`` matters: without it the decoder spends most of
        its capacity learning to emit padding, and the shorter values -- which
        are most of them -- get almost no gradient.
        """
        b, f, t, c = logits.shape
        return tf.cross_entropy(
            logits.reshape(b * f * t, c),
            targets[:, :, 1:].reshape(-1),
            ignore_index=CHAR_PAD,
        )

    @torch.no_grad()
    def decode(self, states: torch.Tensor, mask: torch.Tensor) -> list[list[GenPrediction]]:
        """Greedy decode. ``(B, L, d), (B, L) -> B lists of F predictions``.

        Greedy rather than beam search: a beam would improve the baseline's
        strings but would also multiply its already-dominant decode latency, and
        the efficiency comparison reported in ``docs/RESULTS.md`` uses greedy for
        both, which is the arrangement most favourable to the baseline on time.
        """
        b = states.shape[0]
        pooled = states[:, 0]
        out: list[list[GenPrediction]] = [[] for _ in range(b)]
        for f, name in enumerate(FIELDS):
            fid = torch.full((b,), f, dtype=torch.long, device=states.device)
            hidden = torch.tanh(self.init_h(torch.cat([pooled, self.field_emb(fid)], dim=-1)))
            cur = torch.full((b,), CHAR_BOS, dtype=torch.long, device=states.device)
            done = torch.zeros(b, dtype=torch.bool, device=states.device)
            seqs = [[] for _ in range(b)]
            logps = [[] for _ in range(b)]
            for _ in range(self.max_len - 1):
                logits, hidden = self._step(cur, hidden, states, mask)
                probs = torch.softmax(logits.float(), dim=-1)
                nxt = probs.argmax(dim=-1)
                conf = probs.gather(1, nxt.unsqueeze(1)).squeeze(1)
                for i in range(b):
                    if done[i]:
                        continue
                    token = int(nxt[i])
                    if token in (CHAR_EOS, CHAR_PAD):
                        done[i] = True
                        logps[i].append(float(conf[i]))
                        continue
                    seqs[i].append(token)
                    logps[i].append(float(conf[i]))
                cur = nxt
                if bool(done.all()):
                    break
            for i in range(b):
                text = decode_chars(seqs[i])
                ps = logps[i] or [float("nan")]
                geo = float(np.exp(np.mean(np.log(np.clip(ps, 1e-12, 1.0)))))
                out[i].append(GenPrediction(name, text.strip(), geo))
        return out
