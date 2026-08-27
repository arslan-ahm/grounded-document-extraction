"""The extraction record every method produces, and the candidate interface.

One type, :class:`Extraction`, is what the span head, the generative head, the
heuristic baseline and the LLM baseline all return, so the metrics never branch
on method. Two of its fields carry the whole argument:

``span`` -- the provenance. ``None`` means "this value has no location in the
document", which is the *only* state a generative extractor can be in and is
impossible for a selection extractor that emitted anything.

``grounded`` -- whether the emitted string actually occurs as a contiguous token
span. For selection this is true by construction. For generation it has to be
*checked*, and the fraction of emitted values where it is false is the
hallucination rate.

:class:`Candidate` exists because the verification loop needs somewhere to go
when a check fails. A selection head supplies a ranked candidate set for free
(the top-k of a distribution over spans). A generative head does not: its
"second best string" requires a beam, which is why the generative arms in this
repository can only *abstain* on a failed check and not re-select. That asymmetry
is real and is reported rather than papered over.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from gdx.data.schema import FIELDS, Document, canonical_value, normalise_date


@dataclass
class Candidate:
    """One option for a field: a string, optionally with its provenance span."""

    value: str
    prob: float
    span: tuple[int, int] | None = None

    @property
    def has_provenance(self) -> bool:
        return self.span is not None


@dataclass
class Extraction:
    """The output for one field of one document.

    Attributes:
        field_name: Field from :data:`gdx.data.schema.FIELDS`.
        value: Emitted string. Empty exactly when ``abstained`` is true.
        span: Inclusive token index range the value was read from, or ``None``.
        confidence: Probability in ``[0, 1]``, or ``NaN`` if the method does not
            produce one. Used by the risk-coverage and calibration metrics, which
            drop ``NaN`` rather than substituting a value.
        abstained: The method declined to emit.
        grounded: The emitted string occurs as a contiguous token span.
        checks: Per-check outcome; ``None`` for a check that did not apply.
        n_iters: Verification iterations consumed (0 when verification is off).
        reason: Short machine-readable cause of an abstention.
    """

    field_name: str
    value: str = ""
    span: tuple[int, int] | None = None
    confidence: float = float("nan")
    abstained: bool = True
    grounded: bool = False
    checks: dict[str, bool | None] = field(default_factory=dict)
    n_iters: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("checks")
        out["span_start"] = -1 if self.span is None else self.span[0]
        out["span_end"] = -1 if self.span is None else self.span[1]
        out.pop("span")
        for name in ("type", "provenance", "arithmetic"):
            value = self.checks.get(name)
            out[f"check_{name}"] = "" if value is None else ("pass" if value else "fail")
        return out

    @classmethod
    def abstain(cls, field_name: str, reason: str, confidence: float = float("nan"),
                n_iters: int = 0) -> Extraction:
        """An abstention. Never carries a value, a span or a grounding claim."""
        return cls(
            field_name=field_name,
            value="",
            span=None,
            confidence=confidence,
            abstained=True,
            grounded=False,
            n_iters=n_iters,
            reason=reason,
        )


def mark_grounding(doc: Document, ext: Extraction) -> Extraction:
    """Set ``grounded`` by asking the document whether the value occurs in it.

    Called on *every* extraction from *every* method, including this project's
    own, so the guarantee is measured rather than assumed. A test asserts that
    the measurement never comes back false for a span-selected value; if the
    guarantee were ever broken by a bug, that test -- not a docstring -- is what
    would catch it.
    """
    if ext.abstained or not ext.value:
        ext.grounded = False
        return ext
    ext.grounded = doc.contains_value(ext.field_name, ext.value)
    return ext


def normalise_extraction(ext: Extraction) -> Extraction:
    """Post-hoc deterministic date normalisation on a *grounded* value.

    This is the ``span_verify_norm`` arm. It is included because the honest
    limitation of selection is that "3rd of Jan 2024" can never equal the target
    ``2024-01-03``, and a reader is entitled to ask what it costs to fix that.

    **It weakens the guarantee, and that is stated plainly.** After
    normalisation the emitted string is no longer a contiguous document span; it
    is a *pure deterministic function of one*. Provenance survives -- the span is
    still recorded and still points at real tokens -- but the strict
    "emitted string appears in the document" property does not, so this arm's
    hallucination rate is measured under a provenance-based definition and is
    reported separately from the strict one.
    """
    if ext.abstained or not ext.value:
        return ext
    if ext.field_name not in {"invoice_date", "due_date"}:
        return ext
    iso = normalise_date(ext.value)
    if iso:
        ext.value = iso
    return ext


def empty_extractions(reason: str = "no_prediction") -> dict[str, Extraction]:
    """All fields abstained. The correct output for a document with no tokens."""
    return {name: Extraction.abstain(name, reason) for name in FIELDS}


def value_matches_truth(field_name: str, predicted: str, target: str) -> bool:
    """Canonical equality used by the *lenient* field-accuracy metric."""
    if not predicted or not target:
        return False
    return canonical_value(field_name, predicted) == canonical_value(field_name, target)
