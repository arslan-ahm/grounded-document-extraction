"""Field-level accuracy, coverage and hallucination.

Every metric here is computed from a list of :class:`FieldRecord`, one per
(document, field) pair, and every aggregate reports the count behind it. That is
not decoration: the interesting metrics in this project are conditional
(*"accuracy on fields whose written form requires normalisation"*) and a
conditional mean without its ``n`` is not interpretable.

Three accuracy notions are reported side by side because they answer different
questions and collapsing them would hide this project's central limitation:

``strict`` -- the emitted string equals the target after case and whitespace
normalisation only. This is what a consumer that needs ISO dates requires.

``canonical`` -- equal after parsing dates to ISO and amounts to numbers. This is
the generous reading, and it is where selection does well.

The gap between the two, restricted to fields with
``requires_normalisation=True``, **is** the measured cost of selection-only
extraction.

Abstention is scored as correct exactly when the field is genuinely absent, so a
method cannot buy accuracy by declining to answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gdx.data.schema import FIELDS, Document, canonical_value, normalise_text
from gdx.extract import Extraction


@dataclass
class FieldRecord:
    """One (document, field) outcome, with everything the metrics need."""

    doc_id: int
    field_name: str
    truth_present: bool
    truth_value: str
    predicted_value: str
    abstained: bool
    confidence: float
    grounded: bool
    requires_normalisation: bool
    span_exact: bool
    span_iou: float
    n_iters: int
    reason: str

    @property
    def emitted(self) -> bool:
        return not self.abstained and bool(self.predicted_value)

    @property
    def strict_correct(self) -> bool:
        """Correct under literal (whitespace/case-insensitive) equality."""
        if not self.truth_present:
            return not self.emitted
        if not self.emitted:
            return False
        return normalise_text(self.predicted_value) == normalise_text(self.truth_value)

    @property
    def canonical_correct(self) -> bool:
        """Correct after date and amount parsing."""
        if not self.truth_present:
            return not self.emitted
        if not self.emitted:
            return False
        return canonical_value(self.field_name, self.predicted_value) == canonical_value(
            self.field_name, self.truth_value
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "field": self.field_name,
            "truth_present": self.truth_present,
            "truth_value": self.truth_value,
            "predicted_value": self.predicted_value,
            "abstained": self.abstained,
            "emitted": self.emitted,
            "confidence": self.confidence,
            "grounded": self.grounded,
            "requires_normalisation": self.requires_normalisation,
            "strict_correct": self.strict_correct,
            "canonical_correct": self.canonical_correct,
            "span_exact": self.span_exact,
            "span_iou": self.span_iou,
            "n_iters": self.n_iters,
            "reason": self.reason,
        }


def anls(predicted: str, target: str, threshold: float = 0.5) -> float:
    """Average Normalised Levenshtein Similarity for one pair.

    ANLS (Biten et al., 2019) is the standard document-VQA metric: normalised
    edit similarity, thresholded to zero below ``threshold`` so a near-miss does
    not earn partial credit for being nearly right about the wrong value.
    Reported alongside exact match because exact match is brutal on long values
    and ANLS shows whether a wrong answer was close or nonsense.
    """
    a = normalise_text(predicted)
    b = normalise_text(target)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    similarity = 1.0 - _levenshtein(a, b) / max(len(a), len(b))
    return float(similarity if similarity >= threshold else 0.0)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance, one row at a time (O(min(len)) memory)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def build_records(
    doc: Document,
    extractions: dict[str, Extraction],
    span_exact_fn,  # noqa: ANN001
    span_iou_fn,  # noqa: ANN001
) -> list[FieldRecord]:
    """Turn one document's extractions into per-field records.

    ``span_exact_fn`` and ``span_iou_fn`` are injected rather than imported so
    :mod:`gdx.metrics.grounding` stays the single owner of the grounding
    definition and this module cannot drift from it.
    """
    out: list[FieldRecord] = []
    for name in FIELDS:
        truth = doc.fields[name]
        ext = extractions.get(name)
        if ext is None:
            ext = Extraction.abstain(name, "missing")
        out.append(
            FieldRecord(
                doc_id=doc.doc_id,
                field_name=name,
                truth_present=truth.present,
                truth_value=truth.value,
                predicted_value=ext.value,
                abstained=ext.abstained,
                confidence=ext.confidence,
                grounded=ext.grounded,
                requires_normalisation=truth.requires_normalisation,
                span_exact=span_exact_fn(doc, truth, ext),
                span_iou=span_iou_fn(doc, truth, ext),
                n_iters=ext.n_iters,
                reason=ext.reason,
            )
        )
    return out


def _mean(values: list[float]) -> tuple[float, int]:
    """Mean of the finite entries, and how many there were.

    Returns ``(NaN, 0)`` on an empty selection. An empty selection does not have
    a mean, and returning 0.0 or 1.0 there is how a flattering table gets built.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), 0
    return float(arr.mean()), int(arr.size)


def summarise(records: list[FieldRecord]) -> dict[str, float]:
    """Aggregate field metrics, each accompanied by its ``n_*`` count.

    Definitions worth stating precisely:

    ``coverage`` -- fraction of (document, field) pairs on which the method
    emitted a value, over *all* pairs including those where the field is absent.

    ``hallucination_rate`` -- of the values actually emitted, the fraction that
    do not occur anywhere in the document. Structurally 0 for a selection head;
    the denominator is emissions, not pairs, because a method that abstains on
    everything has no hallucinations and should not be rewarded for it.

    ``precision`` -- of emitted values, the fraction correct. ``recall`` -- of
    genuinely-present fields, the fraction emitted correctly. Both ``NaN`` when
    their denominator is empty.
    """
    if not records:
        return {"n_records": 0}
    emitted = [r for r in records if r.emitted]
    present = [r for r in records if r.truth_present]
    absent = [r for r in records if not r.truth_present]
    needs_norm = [r for r in present if r.requires_normalisation]
    no_norm = [r for r in present if not r.requires_normalisation]

    strict, n_strict = _mean([float(r.strict_correct) for r in records])
    canon, n_canon = _mean([float(r.canonical_correct) for r in records])
    out: dict[str, float] = {
        "n_records": float(len(records)),
        "strict_accuracy": strict,
        "n_strict_accuracy": float(n_strict),
        "canonical_accuracy": canon,
        "n_canonical_accuracy": float(n_canon),
        "coverage": _mean([float(r.emitted) for r in records])[0],
    }

    hall, n_hall = _mean([float(not r.grounded) for r in emitted])
    out["hallucination_rate"] = hall
    out["n_emitted"] = float(n_hall)

    prec, n_prec = _mean([float(r.canonical_correct) for r in emitted])
    out["precision"] = prec
    out["n_precision"] = float(n_prec)

    rec, n_rec = _mean([float(r.canonical_correct) for r in present])
    out["recall"] = rec
    out["n_recall"] = float(n_rec)
    if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0:
        out["f1"] = float(2 * prec * rec / (prec + rec))
    else:
        out["f1"] = float("nan")

    out["anls"], out["n_anls"] = (
        lambda t: (t[0], float(t[1]))
    )(_mean([anls(r.predicted_value, r.truth_value) for r in present]))

    abst, n_abst = _mean([float(r.abstained) for r in absent])
    out["absent_abstain_rate"] = abst
    out["n_absent"] = float(n_abst)
    out["present_emit_rate"] = _mean([float(r.emitted) for r in present])[0]

    strict_nn, n_nn = _mean([float(r.strict_correct) for r in needs_norm])
    out["strict_accuracy_needs_norm"] = strict_nn
    out["n_needs_norm"] = float(n_nn)
    out["canonical_accuracy_needs_norm"] = _mean(
        [float(r.canonical_correct) for r in needs_norm]
    )[0]
    strict_ok, n_ok = _mean([float(r.strict_correct) for r in no_norm])
    out["strict_accuracy_verbatim"] = strict_ok
    out["n_verbatim"] = float(n_ok)
    out["mean_verify_iters"] = _mean([float(r.n_iters) for r in records])[0]
    return out


def summarise_per_field(records: list[FieldRecord]) -> dict[str, dict[str, float]]:
    """The same aggregates, split by field name."""
    out: dict[str, dict[str, float]] = {}
    for name in FIELDS:
        subset = [r for r in records if r.field_name == name]
        if subset:
            out[name] = summarise(subset)
    return out
