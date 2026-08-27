"""The bounded verification loop.

Three checks, then a repair loop with a hard iteration cap.

**Provenance.** Does the emitted string occur as a contiguous token span of the
document? For a span-selected value this is true by construction; for a generated
one it is a real test, and it is the single check that eliminates hallucination.
That the *check* rather than the *head* is what removes hallucinated values is a
finding this repository reports rather than hides -- see
``docs/RESULTS.md`` section on the ``generative_verify`` arm.

**Type.** Could the string be an instance of the field's type at all? A necessary
condition only: it cannot tell a subtotal from a total.

**Arithmetic.** ``subtotal + tax == total``, evaluated on the *parsed* values of
three separately-grounded spans. This is the check that provenance makes possible
and that a bare string output cannot support: three trusted strings can be
inconsistent with each other, and noticing that is free once each one is a real
number read from a real place.

The loop is bounded because an unbounded repair loop is not an algorithm with a
runtime. Each iteration either fixes an inconsistency by taking a field's
next-best candidate or, when a field has no candidates left, abstains on the
fields the failing constraint implicates. ``max_iters`` caps the total, and the
iteration count is recorded per document so the cost is measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

from gdx.config import VerifyConfig
from gdx.data.schema import (
    AMOUNT_FIELDS,
    FIELDS,
    Document,
    normalise_amount,
    type_matches,
)
from gdx.extract import Candidate, Extraction, mark_grounding


@dataclass
class VerificationTrace:
    """What the loop did, for the ablation tables and the notebooks."""

    n_iters: int = 0
    type_rejections: int = 0
    provenance_rejections: int = 0
    arithmetic_rejections: int = 0
    reselections: int = 0
    abstentions: int = 0
    arithmetic_checked: bool = False
    arithmetic_final_ok: bool | None = None

    def to_dict(self) -> dict[str, float | bool | None]:
        return {
            "verify_iters": self.n_iters,
            "verify_type_rejections": self.type_rejections,
            "verify_provenance_rejections": self.provenance_rejections,
            "verify_arithmetic_rejections": self.arithmetic_rejections,
            "verify_reselections": self.reselections,
            "verify_abstentions": self.abstentions,
            "verify_arithmetic_checked": self.arithmetic_checked,
            "verify_arithmetic_ok": self.arithmetic_final_ok,
        }


def provenance_ok(doc: Document, field_name: str, value: str) -> bool:
    """Does ``value`` occur as a contiguous token span of ``doc``?"""
    return bool(value) and doc.contains_value(field_name, value)


def arithmetic_residual(values: dict[str, str]) -> float:
    """``subtotal + tax - total`` from written strings, or ``NaN``.

    ``NaN`` when any of the three is missing or unparseable -- which is the
    common case, because the generator drops ``subtotal`` or ``tax`` on some
    documents. Returning 0.0 there would silently claim the check passed on
    documents where it could not run, inflating the reported consistency rate.
    """
    parsed = {name: normalise_amount(values.get(name, "")) for name in AMOUNT_FIELDS}
    if any(v != v for v in parsed.values()):
        return float("nan")
    return float(parsed["subtotal"] + parsed["tax"] - parsed["total"])


def arithmetic_ok(values: dict[str, str], tol: float) -> bool | None:
    """``True``/``False``, or ``None`` when the check does not apply."""
    residual = arithmetic_residual(values)
    if residual != residual:
        return None
    return abs(residual) <= tol


def _candidate_text(doc: Document, cand: Candidate) -> str:
    """A candidate's surface string: its span's text if it has one."""
    if cand.span is not None:
        return doc.span_text(cand.span[0], cand.span[1])
    return cand.value


def _accept(doc: Document, field_name: str, cand: Candidate, n_iters: int) -> Extraction:
    text = _candidate_text(doc, cand)
    ext = Extraction(
        field_name=field_name,
        value=text,
        span=cand.span,
        confidence=cand.prob,
        abstained=not text,
        grounded=False,
        n_iters=n_iters,
    )
    ext.checks["type"] = type_matches(field_name, text) if text else None
    ext.checks["provenance"] = provenance_ok(doc, field_name, text) if text else None
    return mark_grounding(doc, ext)


def verify_document(
    doc: Document,
    candidates: dict[str, list[Candidate]],
    cfg: VerifyConfig,
) -> tuple[dict[str, Extraction], VerificationTrace]:
    """Run the bounded verification loop over one document's candidate sets.

    Args:
        doc: The source document.
        candidates: Per field, a ranked list of :class:`Candidate`. An empty list
            means the method produced nothing, which becomes an abstention.
        cfg: Verification configuration. With ``enabled=False`` the top candidate
            is emitted unchecked, which is the ``span_only`` ablation.

    Returns:
        Per-field extractions, and a :class:`VerificationTrace`.
    """
    trace = VerificationTrace()
    out: dict[str, Extraction] = {}
    cursor: dict[str, int] = dict.fromkeys(FIELDS, 0)

    # --- stage 1: per-field local checks -----------------------------------
    for name in FIELDS:
        options = candidates.get(name) or []
        if not options:
            out[name] = Extraction.abstain(name, "no_candidate")
            continue
        if not cfg.enabled:
            out[name] = _accept(doc, name, options[0], 0)
            continue
        chosen: Extraction | None = None
        limit = min(len(options), max(1, cfg.max_iters))
        for k in range(limit):
            cand = options[k]
            text = _candidate_text(doc, cand)
            if not text:
                chosen = Extraction.abstain(name, "empty_candidate", cand.prob, k)
                break
            if cfg.type_check and not type_matches(name, text):
                trace.type_rejections += 1
                trace.n_iters += 1
                continue
            if not provenance_ok(doc, name, text):
                trace.provenance_rejections += 1
                trace.n_iters += 1
                continue
            if k > 0:
                trace.reselections += 1
            chosen = _accept(doc, name, cand, k)
            cursor[name] = k
            break
        if chosen is None:
            trace.abstentions += 1
            conf = options[0].prob
            chosen = (
                Extraction.abstain(name, "checks_failed", conf, limit)
                if cfg.abstain
                else _accept(doc, name, options[0], limit)
            )
            if not cfg.abstain:
                chosen.reason = "checks_failed_emitted_anyway"
            cursor[name] = 0
        out[name] = chosen

    if cfg.enabled and cfg.min_confidence > 0.0:
        for name, ext in out.items():
            low = ext.confidence == ext.confidence and ext.confidence < cfg.min_confidence
            if not ext.abstained and low and cfg.abstain:
                trace.abstentions += 1
                out[name] = Extraction.abstain(
                    name, "low_confidence", ext.confidence, ext.n_iters
                )

    # --- stage 2: the global arithmetic constraint --------------------------
    if cfg.enabled and cfg.arithmetic:
        _repair_arithmetic(doc, candidates, cursor, out, cfg, trace)

    return out, trace


def _repair_arithmetic(
    doc: Document,
    candidates: dict[str, list[Candidate]],
    cursor: dict[str, int],
    out: dict[str, Extraction],
    cfg: VerifyConfig,
    trace: VerificationTrace,
) -> None:
    """Bounded repair of ``subtotal + tax == total``.

    Each iteration advances the *least confident* participating field to its
    next-best candidate that passes the local checks. When no participant has a
    candidate left, the loop stops and abstains on all three (or emits them
    marked failed, if ``abstain`` is off). The cap is ``cfg.max_iters``, so the
    worst case is three candidate advances, not a search.
    """
    values = {n: out[n].value for n in AMOUNT_FIELDS}
    status = arithmetic_ok(values, cfg.arithmetic_tol)
    if status is None:
        for name in AMOUNT_FIELDS:
            out[name].checks["arithmetic"] = None
        return
    trace.arithmetic_checked = True

    for _ in range(max(1, cfg.max_iters)):
        if status:
            break
        trace.arithmetic_rejections += 1
        trace.n_iters += 1
        order = sorted(
            AMOUNT_FIELDS,
            key=lambda n: (
                out[n].confidence if out[n].confidence == out[n].confidence else -1.0
            ),
        )
        advanced = False
        for name in order:
            options = candidates.get(name) or []
            nxt = cursor[name] + 1
            while nxt < len(options):
                text = _candidate_text(doc, options[nxt])
                if text and type_matches(name, text) and provenance_ok(doc, name, text):
                    cursor[name] = nxt
                    out[name] = _accept(doc, name, options[nxt], out[name].n_iters + 1)
                    trace.reselections += 1
                    advanced = True
                    break
                nxt += 1
            if advanced:
                break
        if not advanced:
            break
        values = {n: out[n].value for n in AMOUNT_FIELDS}
        status = arithmetic_ok(values, cfg.arithmetic_tol)

    trace.arithmetic_final_ok = bool(status)
    for name in AMOUNT_FIELDS:
        out[name].checks["arithmetic"] = bool(status)
    if not status:
        for name in AMOUNT_FIELDS:
            if cfg.abstain:
                trace.abstentions += 1
                out[name] = Extraction.abstain(
                    name, "arithmetic_failed", out[name].confidence, out[name].n_iters
                )
                out[name].checks["arithmetic"] = False
            else:
                out[name].reason = "arithmetic_failed_emitted_anyway"
