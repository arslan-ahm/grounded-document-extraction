"""Document representation, field schema, and the value normalisers.

The type that carries the whole argument of this repository is
:class:`FieldTruth`. It records not only the target string but **the token index
range it was written at**, because the generator placed it there. That makes
grounding accuracy measurable against an exact oracle rather than eyeballed --
no human-annotated extraction dataset records where a value was read from, which
is why the shipped data is generated.

:attr:`FieldTruth.requires_normalisation` is the other load-bearing flag. When a
date is written "3rd of Jan 2024" and the target is ``2024-01-03``, no
contiguous span of document tokens equals the target. A selection-only extractor
**cannot** be right there. That is a structural limitation of the ideology, and
this flag is what lets the results split accuracy by whether the limitation
applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Field schema
# ---------------------------------------------------------------------------

#: The extraction targets, in a fixed order that indexes every model head.
FIELDS: tuple[str, ...] = (
    "invoice_id",
    "invoice_date",
    "due_date",
    "vendor_name",
    "po_number",
    "subtotal",
    "tax",
    "total",
)

FIELD_INDEX: dict[str, int] = {name: i for i, name in enumerate(FIELDS)}

#: Fields whose values are monetary amounts and therefore enter the arithmetic
#: consistency check ``subtotal + tax == total``.
AMOUNT_FIELDS: tuple[str, ...] = ("subtotal", "tax", "total")

#: Fields whose targets are ISO dates.
DATE_FIELDS: tuple[str, ...] = ("invoice_date", "due_date")

#: Per-field type patterns for the verification loop's type check. A candidate
#: span whose text cannot possibly be an instance of the field's type is
#: rejected before it is ever emitted.
TYPE_PATTERNS: dict[str, re.Pattern[str]] = {
    "invoice_id": re.compile(r"^[A-Z]{2,4}[-/ ]?\d{3,8}$", re.I),
    "vendor_name": re.compile(r"^[A-Za-z][A-Za-z&.,'\- ]{2,}$"),
    "po_number": re.compile(r"^(PO|P\.O\.|ORD)[-/ ]?\d{3,8}$", re.I),
}

#: Currency prefixes stripped before an amount is parsed. Uppercased for
#: matching; the symbol forms are single characters so case is irrelevant there.
CURRENCY_TOKENS: tuple[str, ...] = ("EUR", "GBP", "USD", "$", "€", "£")

#: A written amount, after the currency prefix has been removed: either a plain
#: integer/decimal or a comma-grouped one. Anything else is not an amount.
_STRICT_AMOUNT = re.compile(r"^(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?$")

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_ORDINAL = re.compile(r"^(\d{1,2})(st|nd|rd|th)$", re.I)

#: Words a written date may contain without ceasing to be only a date.
_CONNECTORS = frozenset({"of", "the", "on"})


# ---------------------------------------------------------------------------
# Normalisers
# ---------------------------------------------------------------------------


def normalise_text(value: str) -> str:
    """Case- and whitespace-insensitive form used for string comparison.

    Deliberately conservative: it collapses whitespace, lowercases, and strips
    trailing punctuation. It does **not** reformat dates or amounts, because
    doing so inside the equality test would hide the very limitation this
    project is measuring.
    """
    if value is None:
        return ""
    out = re.sub(r"\s+", " ", str(value)).strip().lower()
    return out.strip(" .,;:")


def normalise_amount(value: str) -> float:
    """Parse a written amount to a float, or ``NaN`` if it is not one.

    Handles ``$1,234.56``, ``1234.56``, ``EUR 1234.56`` and parenthesised
    negatives ``(45.00)``.

    **Strictness is the point.** An earlier version stripped every non-digit
    character, which made ``"Net Amount EUR 30,614.90"`` parse as ``30614.90``.
    That silently turned label text into a valid amount, inflating the count of
    document spans that "contain" a value and therefore weakening the
    hallucination metric in this method's favour. The parser now *rejects*
    anything with residual letters or a second numeric group.
    """
    if value is None:
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    for symbol in CURRENCY_TOKENS:
        if text.upper().startswith(symbol):
            text = text[len(symbol) :].strip()
            break
    if text.startswith("-"):
        negative = True
        text = text[1:].strip()
    if not _STRICT_AMOUNT.match(text):
        return float("nan")
    try:
        out = float(text.replace(",", ""))
    except ValueError:
        return float("nan")
    return -out if negative else out


def type_matches(field_name: str, text: str) -> bool:
    """Could ``text`` be an instance of ``field_name``'s type?

    This is the verification loop's *type check*: a candidate span whose surface
    form cannot be an instance of the field's type is rejected before it is
    emitted. It is deliberately a necessary condition and not a sufficient one --
    it cannot tell a subtotal from a total, which is what the arithmetic check is
    for.
    """
    text = (text or "").strip()
    if not text:
        return False
    if field_name in AMOUNT_FIELDS:
        parsed = normalise_amount(text)
        return parsed == parsed
    if field_name in DATE_FIELDS:
        return bool(normalise_date(text))
    pattern = TYPE_PATTERNS.get(field_name)
    return bool(pattern.match(text)) if pattern is not None else True


def normalise_date(value: str) -> str:
    """Best-effort ISO ``YYYY-MM-DD`` for a written date, else ``""``.

    This is a *deterministic post-processor*, not a model. It exists so the
    repository can quantify the value of adding normalisation on top of grounded
    selection -- see the ``span_verify_norm`` arm -- while being explicit that
    doing so weakens the "emitted string is a document substring" guarantee to
    "emitted string is a pure function of a grounded span".
    """
    if not value:
        return ""
    text = re.sub(r"[,]", " ", str(value)).strip()
    iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", text)
    if iso:
        y, m, d = (int(g) for g in iso.groups())
        return _iso(y, m, d)
    slash = re.match(r"^(\d{1,2})[/.](\d{1,2})[/.](\d{2,4})$", text)
    if slash:
        a, b, c = (int(g) for g in slash.groups())
        year = c if c > 99 else 2000 + c
        # The generator writes day/month/year; stated so the convention is not
        # silently assumed by a reader.
        return _iso(year, b, a)
    words = [w for w in text.split() if w]
    day = month = year = None
    for word in words:
        low = word.lower().strip(".")
        ordinal = _ORDINAL.match(low)
        if ordinal:
            day = int(ordinal.group(1))
            continue
        if low in _MONTHS:
            month = _MONTHS[low]
            continue
        if low.isdigit():
            n = int(low)
            if n > 31:
                year = n if n > 99 else 2000 + n
                continue
            if day is None:
                day = n
                continue
            return ""
        if low not in _CONNECTORS:
            # Any unrecognised word means this span is not *only* a date. An
            # earlier version ignored stray words, which made "Issued 29th of
            # March 2024" parse to the same ISO date as "29th of March 2024" and
            # so counted the label token as part of a correct grounding.
            return ""
    if day and month and year:
        return _iso(year, month, day)
    return ""


def _iso(year: int, month: int, day: int) -> str:
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d}"


def canonical_value(field_name: str, value: str) -> str:
    """The comparison key for a field: dates go to ISO, everything else to text.

    Amounts are compared numerically elsewhere; here they are normalised as
    text with the currency symbol and thousands separators removed so that
    ``$1,234.56`` and ``1234.56`` are the same answer.
    """
    if field_name in DATE_FIELDS:
        iso = normalise_date(value)
        return iso or normalise_text(value)
    if field_name in AMOUNT_FIELDS:
        parsed = normalise_amount(value)
        if parsed == parsed:  # not NaN
            return f"{parsed:.2f}"
        return normalise_text(value)
    return normalise_text(value)


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    """One OCR-style token: a string and where it sits on which page.

    Boxes are ``(x0, y0, x1, y1)`` in normalised page coordinates on ``[0, 1]``,
    with ``y`` increasing downwards, which is the convention every document
    layout model uses.
    """

    text: str
    box: tuple[float, float, float, float]
    page: int = 0

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class FieldTruth:
    """Ground truth for one field, including exact provenance.

    Attributes:
        name: Field name from :data:`FIELDS`.
        value: The *target* string an extractor is scored against.
        present: ``False`` when the field is absent from the document, in which
            case abstaining is the correct behaviour and emitting anything is
            wrong.
        span: Inclusive ``(start, end)`` token indices the value was written at,
            or ``None`` when absent.
        span_text: The verbatim document text of that span.
        requires_normalisation: ``True`` when ``span_text`` does not literally
            equal ``value`` -- the case a selection-only extractor cannot win.
        alt_spans: Every *other* span in the document carrying the same value.
            Invoices repeat values ("Total Due" and "Amount Due" both print the
            total, and the invoice id reprints on every continuation page), so a
            grounding metric that insisted on one canonical location would
            penalise a correct read. Any span in ``[span] + alt_spans`` counts.
    """

    name: str
    value: str = ""
    present: bool = False
    span: tuple[int, int] | None = None
    span_text: str = ""
    requires_normalisation: bool = False
    alt_spans: list[tuple[int, int]] = field(default_factory=list)

    @property
    def all_spans(self) -> list[tuple[int, int]]:
        """The primary span followed by every equivalent occurrence."""
        return ([] if self.span is None else [self.span]) + list(self.alt_spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "present": self.present,
            "span_start": -1 if self.span is None else self.span[0],
            "span_end": -1 if self.span is None else self.span[1],
            "span_text": self.span_text,
            "requires_normalisation": self.requires_normalisation,
            "n_occurrences": len(self.all_spans),
        }


@dataclass
class Document:
    """A synthetic visually-rich document with exact provenance ground truth."""

    doc_id: int
    tokens: list[Token] = field(default_factory=list)
    fields: dict[str, FieldTruth] = field(default_factory=dict)
    n_pages: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def texts(self) -> list[str]:
        return [t.text for t in self.tokens]

    def span_text(self, start: int, end: int) -> str:
        """Verbatim text of the inclusive token range ``[start, end]``.

        Returns ``""`` for an out-of-range or inverted range rather than raising,
        because a model may propose one and the caller must be able to score it.
        """
        if start < 0 or end < start or end >= len(self.tokens):
            return ""
        return " ".join(t.text for t in self.tokens[start : end + 1])

    def union_box(self, start: int, end: int) -> tuple[float, float, float, float] | None:
        """Axis-aligned union of the boxes in ``[start, end]``, or ``None``.

        ``None`` when the range is invalid, or when it straddles two pages --
        a box union across pages is not a region and must not be scored as one.
        """
        if start < 0 or end < start or end >= len(self.tokens):
            return None
        toks = self.tokens[start : end + 1]
        pages = {t.page for t in toks}
        if len(pages) != 1:
            return None
        return (
            min(t.box[0] for t in toks),
            min(t.box[1] for t in toks),
            max(t.box[2] for t in toks),
            max(t.box[3] for t in toks),
        )

    def contains_value(self, field_name: str, value: str, max_span_len: int = 8) -> bool:
        """Does ``value`` appear as some contiguous token span of this document?

        This is the predicate behind the hallucination metric. It is evaluated
        under :func:`canonical_value` so that a formatting difference does not
        count as a hallucination -- the question is whether the *content* was
        present, not whether the whitespace matched.
        """
        target = canonical_value(field_name, value)
        if not target:
            return False
        n = len(self.tokens)
        for start in range(n):
            page = self.tokens[start].page
            for end in range(start, min(start + max_span_len, n)):
                if self.tokens[end].page != page:
                    break
                if canonical_value(field_name, self.span_text(start, end)) == target:
                    return True
        return False
