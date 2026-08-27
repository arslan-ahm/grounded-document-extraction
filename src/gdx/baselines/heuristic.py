"""Rule-based extraction: label matching plus geometric nearest-value search.

**This baseline is strong and is given a fair fight.** Label-and-geometry rules
are what production invoice extraction actually used before end-to-end models,
and a strawman version -- one label string per field, nearest number anywhere on
the page -- would make the neural comparison worthless. So:

* it matches against *every* label synonym the generator can emit, from the same
  shared table (:data:`gdx.data.lexicon.LABEL_SYNONYMS`), including multi-word
  labels;
* it knows the field types and only accepts a value that could be an instance of
  the field's type;
* it searches to the right of the label first and then below it, with a
  configurable weighting, because invoice values sit in both places;
* the search geometry is a **swept configuration** (:data:`VARIANTS`) and the
  winner is chosen on the *validation* split, never on test.

It also inherits the provenance guarantee for free: it returns token spans, so
it structurally cannot hallucinate either. That is worth saying out loud -- the
grounding property is a property of *selection*, not of neural networks, and the
comparison this repository cares about is selection versus generation, not
learned versus rule-based.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from gdx.data.lexicon import LABEL_SYNONYMS
from gdx.data.schema import FIELDS, Document, type_matches
from gdx.extract import Candidate


@dataclass(frozen=True)
class HeuristicVariant:
    """A geometric search configuration for the rule baseline.

    Two placements are supported because invoices use both: a value on the
    *same row* as its label (the meta grid and the summary block) and a value on
    the *line below* it (the vendor block under "Supplier:"). A variant that
    only searched to the right would miss the whole second family, and reporting
    it as the rule baseline's ceiling would be a strawman.

    Attributes:
        row_tol: Vertical centre distance within which two tokens count as the
            same row. Must exceed the generator's box jitter plus the page-skew
            displacement, or rotation alone breaks row membership.
        max_dx_row: Horizontal reach for a same-row value.
        max_dy_below: Vertical reach for a below-label value.
        max_dx_below: Horizontal offset tolerated for a below-label value.
        below_penalty: Constant added to a below-label distance. ``inf`` disables
            the below placement entirely, which is one of the swept variants.
        max_span: Maximum tokens a value may span.
    """

    name: str
    row_tol: float
    max_dx_row: float
    max_dy_below: float
    max_dx_below: float
    below_penalty: float
    max_span: int
    same_page_only: bool = True
    prefer_last: bool = False


VARIANTS: tuple[HeuristicVariant, ...] = (
    HeuristicVariant("row_only", 0.018, 0.60, 0.0, 0.0, float("inf"), 4),
    HeuristicVariant("row_first", 0.018, 0.60, 0.045, 0.12, 0.50, 4),
    HeuristicVariant("row_first_wide", 0.024, 0.90, 0.060, 0.20, 0.50, 5),
    HeuristicVariant("below_cheap", 0.018, 0.60, 0.060, 0.20, 0.05, 5),
    HeuristicVariant("below_cheap_last", 0.018, 0.60, 0.060, 0.20, 0.05, 5, prefer_last=True),
    HeuristicVariant("any_page", 0.024, 0.90, 0.060, 0.20, 0.30, 5, same_page_only=False),
)

VARIANT_BY_NAME = {v.name: v for v in VARIANTS}

_WORD = re.compile(r"[A-Za-z]+")


def _norm_word(text: str) -> str:
    """Letters only, lowercased -- so "Total:" and "TOTAL" match one label."""
    return "".join(_WORD.findall(text)).lower()


def find_label_positions(doc: Document, field_name: str) -> list[tuple[int, int]]:
    """Every token range in ``doc`` that spells one of the field's labels.

    Longer synonyms are matched first, so "Total Due" wins over the bare "Total"
    it contains. Without that, the label anchor for ``total`` would land one
    token early and the geometric search would start from the wrong place --
    which is precisely the kind of quiet bug that makes a rule baseline look
    worse than it is.
    """
    words = [_norm_word(t) for t in doc.texts]
    synonyms = sorted(
        ({_norm_word(s) for s in LABEL_SYNONYMS[field_name]}),
        key=len,
        reverse=True,
    )
    hits: list[tuple[int, int]] = []
    n = len(words)
    for i in range(n):
        for syn in synonyms:
            if not syn:
                continue
            joined = ""
            matched = -1
            for j in range(i, min(i + 4, n)):
                joined += words[j]
                if joined == syn:
                    matched = j
                    break
                if not syn.startswith(joined):
                    break
            if matched >= 0:
                # Only a *match* ends the synonym search. An earlier version
                # broke out of the synonym loop on the first mismatch too, which
                # meant only the longest synonym was ever tried and the baseline
                # found labels on 20% of fields instead of 96%.
                hits.append((i, matched))
                break
    return hits


def _distance(doc: Document, label_end: int, cand_start: int, v: HeuristicVariant) -> float:
    """Geometric distance from a label anchor to a candidate value start.

    Returns ``inf`` for anything outside the variant's windows, so an
    out-of-range token is excluded rather than merely down-weighted.
    """
    lab = doc.tokens[label_end]
    tok = doc.tokens[cand_start]
    if v.same_page_only and lab.page != tok.page:
        return float("inf")
    lx0, ly0, lx1, ly1 = lab.box
    tx0, ty0, _, ty1 = tok.box
    dy = abs(0.5 * (ty0 + ty1) - 0.5 * (ly0 + ly1))

    if dy <= v.row_tol:
        dx = tx0 - lx1
        if dx < -0.02 or dx > v.max_dx_row:
            return float("inf")
        return max(0.0, dx)

    if v.below_penalty == float("inf"):
        return float("inf")
    dv = ty0 - ly1
    dh = abs(tx0 - lx0)
    if dv < -0.005 or dv > v.max_dy_below or dh > v.max_dx_below:
        return float("inf")
    return v.below_penalty + max(0.0, dv) * 4.0 + dh


def extract_field(
    doc: Document, field_name: str, v: HeuristicVariant, top_k: int = 5
) -> list[Candidate]:
    """Ranked candidates for one field: nearest type-valid span to a label.

    Probabilities are ``1 / (1 + rank)`` renormalised -- an ordinal confidence,
    not a calibrated one. The calibration tables report this baseline's ECE as
    what it is: a rank heuristic dressed as a probability, and it scores badly,
    which is the honest outcome.
    """
    labels = find_label_positions(doc, field_name)
    if not labels:
        return []
    if v.prefer_last:
        labels = labels[::-1]
    n = len(doc.tokens)
    scored: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for _, label_end in labels:
        for start in range(label_end + 1, n):
            dist = _distance(doc, label_end, start, v)
            if dist == float("inf"):
                continue
            for end in range(start, min(start + v.max_span, n)):
                if doc.tokens[end].page != doc.tokens[start].page:
                    break
                text = doc.span_text(start, end)
                if not type_matches(field_name, text):
                    continue
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                scored.append((dist + 0.001 * (end - start), start, end))
    if not scored:
        return []
    scored.sort()
    weights = [1.0 / (1.0 + i) for i in range(min(top_k, len(scored)))]
    total = sum(weights)
    return [
        Candidate(
            value=doc.span_text(s, e),
            prob=w / total,
            span=(s, e),
        )
        for w, (_, s, e) in zip(weights, scored[:top_k], strict=False)
    ]


def extract_candidates(
    doc: Document, variant: HeuristicVariant, top_k: int = 5
) -> dict[str, list[Candidate]]:
    """Candidates for every field of one document."""
    return {name: extract_field(doc, name, variant, top_k) for name in FIELDS}
