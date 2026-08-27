"""Grounding accuracy against the exact provenance oracle.

This is the metric no real extraction benchmark can compute. FUNSD, CORD and
SROIE annotate *values*; none of them records which token the value was read
from, so "did the model look in the right place" is unanswerable there. The
generator places every token, so here it is answerable exactly, and that is the
reason the shipped results are synthetic.

Two definitions, reported separately:

**Exact span match** -- the predicted ``(start, end)`` is one of the truth's
occurrences. Duplicates count: a total printed under both "Total Due" and
"Amount Due" is correctly read from either, so the truth carries
``alt_spans`` and any of them is accepted (see
:attr:`gdx.data.schema.FieldTruth.alt_spans`).

**Box IoU** -- intersection over union of the predicted span's union box with the
best-matching truth occurrence's box. This is the softer measure: a span that
picks up the currency symbol as well as the digits scores a high IoU and a failed
exact match, and the gap between the two tells a reader which kind of error is
happening.

Both are ``NaN`` where undefined -- an absent field has no location, and a method
that abstained made no location claim. Scoring either as 0.0 would conflate
"correctly declined" with "looked in the wrong place".
"""

from __future__ import annotations

import numpy as np

from gdx.data.layout import iou
from gdx.data.schema import Document, FieldTruth
from gdx.extract import Extraction


def span_exact(doc: Document, truth: FieldTruth, ext: Extraction) -> bool:
    """Is the predicted span one of the truth's equivalent occurrences?"""
    del doc
    if not truth.present or ext.span is None:
        return False
    return tuple(ext.span) in {tuple(s) for s in truth.all_spans}


def span_iou(doc: Document, truth: FieldTruth, ext: Extraction) -> float:
    """Best IoU between the predicted box and any truth occurrence's box.

    Returns ``NaN`` when the field is absent from the document (no truth box) or
    the method made no span claim -- including every prediction from a generative
    head, which is exactly the point: generation cannot be scored on grounding
    at all, and the table shows that as ``n/a`` with a count of 0.
    """
    if not truth.present or ext.span is None:
        return float("nan")
    pred_box = doc.union_box(ext.span[0], ext.span[1])
    if pred_box is None:
        return float("nan")
    best = float("nan")
    for span in truth.all_spans:
        truth_box = doc.union_box(span[0], span[1])
        value = iou(pred_box, truth_box)
        if np.isfinite(value) and (not np.isfinite(best) or value > best):
            best = value
    return float(best)


def token_overlap(truth: FieldTruth, ext: Extraction) -> float:
    """Token-index IoU between the predicted span and the nearest truth span.

    A discrete companion to the box IoU. It is insensitive to the geometry noise
    the generator injects, so a disagreement between the two tells a reader that
    box jitter -- not selection -- is responsible.
    """
    if not truth.present or ext.span is None:
        return float("nan")
    pred = set(range(ext.span[0], ext.span[1] + 1))
    best = 0.0
    for span in truth.all_spans:
        gold = set(range(span[0], span[1] + 1))
        union = pred | gold
        if union:
            best = max(best, len(pred & gold) / len(union))
    return float(best)


def summarise_grounding(
    docs: list[Document], per_doc: list[dict[str, Extraction]]
) -> dict[str, float]:
    """Aggregate grounding metrics over a split.

    Args:
        docs: The documents, in order.
        per_doc: Per-document field-name to :class:`Extraction`.

    Returns:
        Means with their contributing counts. ``grounding_exact`` is over the
        (document, field) pairs where the field is *present and the method
        emitted a span*; a method that never emits a span gets ``NaN`` and
        ``n=0``, not 0.0.
    """
    exact: list[float] = []
    ious: list[float] = []
    overlaps: list[float] = []
    claimed = 0
    claimable = 0
    for doc, extractions in zip(docs, per_doc, strict=True):
        for name, truth in doc.fields.items():
            ext = extractions.get(name)
            if ext is None:
                continue
            if truth.present:
                claimable += 1
            if truth.present and ext.span is not None:
                claimed += 1
                exact.append(float(span_exact(doc, truth, ext)))
                ious.append(span_iou(doc, truth, ext))
                overlaps.append(token_overlap(truth, ext))

    def agg(values: list[float]) -> tuple[float, int]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), 0
        return float(arr.mean()), int(arr.size)

    exact_mean, n_exact = agg(exact)
    iou_mean, n_iou = agg(ious)
    ov_mean, n_ov = agg(overlaps)
    return {
        "grounding_exact": exact_mean,
        "n_grounding_exact": float(n_exact),
        "grounding_iou": iou_mean,
        "n_grounding_iou": float(n_iou),
        "grounding_token_iou": ov_mean,
        "n_grounding_token_iou": float(n_ov),
        "provenance_claim_rate": (claimed / claimable) if claimable else float("nan"),
        "n_claimable": float(claimable),
    }
