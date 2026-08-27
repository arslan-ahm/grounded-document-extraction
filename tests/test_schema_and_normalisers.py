"""The value normalisers and the document type, tested against closed forms.

These functions decide what counts as a correct answer and what counts as a
hallucination, so a bug here silently rewrites every number in the repository.
Both known failures found during development are pinned as regression tests:
``normalise_amount`` parsing label text, and ``normalise_date`` ignoring stray
words.
"""

from __future__ import annotations

import math

import pytest

from gdx.data.schema import (
    AMOUNT_FIELDS,
    DATE_FIELDS,
    FIELD_INDEX,
    FIELDS,
    MAX_VALUE_SPAN,
    Document,
    FieldTruth,
    Token,
    canonical_value,
    normalise_amount,
    normalise_date,
    normalise_text,
    type_matches,
)


# --- normalise_text --------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Total  Due ", "total due"),
        ("INV-123", "inv-123"),
        ("Acme Ltd.", "acme ltd"),
        ("a,b;", "a,b"),
        ("", ""),
        (None, ""),
        ("Multi\nline\ttext", "multi line text"),
    ],
)
def test_normalise_text_cases(raw, expected):  # noqa: ANN001
    assert normalise_text(raw) == expected


# --- normalise_amount -----------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.56", 1234.56),
        ("1234.56", 1234.56),
        ("EUR 30,614.90", 30614.90),
        ("GBP 5", 5.0),
        ("(45.00)", -45.0),
        ("-12.30", -12.30),
        ("0.05", 0.05),
        ("1,000", 1000.0),
    ],
)
def test_normalise_amount_parses(raw, expected):  # noqa: ANN001
    assert normalise_amount(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        "Net Amount EUR 30,614.90",   # regression: label text must not parse
        "36,737.88 Registered",
        "Total",
        "12,34.5",                     # malformed thousands group
        "1,2345",
        "",
        None,
        "abc",
        "1.2.3",
        "EUR",
    ],
)
def test_normalise_amount_rejects(raw):  # noqa: ANN001
    assert math.isnan(normalise_amount(raw))


# --- normalise_date -------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-01-03", "2024-01-03"),
        ("03/01/2024", "2024-01-03"),
        ("3rd of Jan 2024", "2024-01-03"),
        ("Jan 3, 2024", "2024-01-03"),
        ("29th of March 2024", "2024-03-29"),
        ("1st of December 2023", "2023-12-01"),
        ("31.12.2024", "2024-12-31"),
    ],
)
def test_normalise_date_parses(raw, expected):  # noqa: ANN001
    assert normalise_date(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "Issued 29th of March 2024",   # regression: stray words must reject
        "29th of March 2024 Due",
        "Due By 2024-01-03",
        "2024-01-03 Due",
        "nonsense",
        "",
        "Jan",
        "2024",
        "2024-13-01",                   # impossible month
        "2024-01-40",                   # impossible day
    ],
)
def test_normalise_date_rejects(raw):  # noqa: ANN001
    assert normalise_date(raw) == ""


def test_date_slash_convention_is_day_month_year():
    """The generator writes ``dd/mm/yyyy``; the parser must agree.

    Getting this backwards would make every slash-form date wrong in a way that
    looked like a model failure rather than a parser one.
    """
    assert normalise_date("03/01/2024") == "2024-01-03"
    assert normalise_date("13/01/2024") == "2024-01-13"


# --- type_matches ---------------------------------------------------------

@pytest.mark.parametrize(
    ("field_name", "text", "expected"),
    [
        ("total", "EUR 5.00", True),
        ("total", "Total", False),
        ("total", "", False),
        ("subtotal", "1,234.00", True),
        ("tax", "Balance", False),
        ("invoice_id", "INV-01234", True),
        ("invoice_id", "ORD-1", False),
        ("invoice_id", "INV01234", True),
        ("po_number", "PO-00123", True),
        ("po_number", "INV-00123", False),
        ("invoice_date", "Jan 3, 2024", True),
        ("invoice_date", "Bracket", False),
        ("vendor_name", "Northwind Systems Ltd", True),
        ("vendor_name", "12345", False),
    ],
)
def test_type_matches_cases(field_name, text, expected):  # noqa: ANN001
    assert type_matches(field_name, text) is expected


def test_type_check_is_necessary_not_sufficient():
    """It cannot tell a subtotal from a total; that is the arithmetic check's job.

    Stated as a test so the limitation is not later mistaken for a bug.
    """
    assert type_matches("subtotal", "100.00")
    assert type_matches("total", "100.00")


# --- canonical_value ------------------------------------------------------

def test_canonical_value_makes_currency_styles_equal():
    for a, b in (("$1,234.56", "1234.56"), ("EUR 5.00", "5"), ("GBP 12.30", "12.3")):
        assert canonical_value("total", a) == canonical_value("total", b)


def test_canonical_value_makes_date_formats_equal():
    forms = ["2024-01-03", "03/01/2024", "3rd of Jan 2024", "Jan 3, 2024"]
    keys = {canonical_value("invoice_date", f) for f in forms}
    assert keys == {"2024-01-03"}


def test_canonical_value_keeps_distinct_values_distinct():
    assert canonical_value("total", "100.00") != canonical_value("total", "100.01")
    assert canonical_value("invoice_id", "INV-1") != canonical_value("invoice_id", "INV-2")


def test_canonical_value_of_unparseable_amount_falls_back_to_text():
    assert canonical_value("total", "Total Due") == "total due"


# --- Document -------------------------------------------------------------

def _doc(texts, pages=None):  # noqa: ANN001, ANN202
    pages = pages or [0] * len(texts)
    tokens = [
        Token(text=t, box=(0.1 * i, 0.1, 0.1 * i + 0.05, 0.13), page=p)
        for i, (t, p) in enumerate(zip(texts, pages, strict=True))
    ]
    return Document(doc_id=0, tokens=tokens, fields={n: FieldTruth(n) for n in FIELDS})


def test_span_text_out_of_range_returns_empty():
    doc = _doc(["a", "b", "c"])
    assert doc.span_text(-1, 1) == ""
    assert doc.span_text(2, 1) == ""
    assert doc.span_text(0, 99) == ""
    assert doc.span_text(0, 2) == "a b c"


def test_union_box_refuses_a_cross_page_range():
    """A box union across pages is not a region and must be ``None``.

    This is the geometric counterpart of the page-confinement rule in span
    decoding, and it is why that rule is necessary.
    """
    doc = _doc(["a", "b"], pages=[0, 1])
    assert doc.union_box(0, 1) is None
    assert doc.union_box(0, 0) is not None


def test_union_box_is_the_axis_aligned_hull():
    doc = _doc(["a", "b", "c"])
    box = doc.union_box(0, 2)
    assert box == (0.0, 0.1, 0.25, 0.13)


def test_value_index_is_cached_and_page_confined():
    doc = _doc(["INV-1", "INV-2"], pages=[0, 1])
    first = doc.value_index("invoice_id")
    assert doc.value_index("invoice_id") is first
    assert "inv-1 inv-2" not in first


def test_contains_value_respects_max_value_span():
    texts = [f"w{i}" for i in range(MAX_VALUE_SPAN + 3)]
    doc = _doc(texts)
    short = " ".join(texts[:MAX_VALUE_SPAN])
    long = " ".join(texts)
    assert doc.contains_value("vendor_name", short)
    assert not doc.contains_value("vendor_name", long)


def test_contains_value_of_empty_is_false():
    doc = _doc(["a"])
    assert not doc.contains_value("total", "")
    assert not doc.contains_value("total", "not-a-number")


def test_empty_document_contains_nothing():
    doc = _doc([])
    assert len(doc) == 0
    assert doc.value_index("total") == frozenset()
    assert not doc.contains_value("total", "1.00")
    assert doc.span_text(0, 0) == ""
    assert doc.union_box(0, 0) is None


def test_single_token_document():
    doc = _doc(["500.00"])
    assert doc.contains_value("total", "500.00")
    assert doc.contains_value("total", "$500")
    assert doc.span_text(0, 0) == "500.00"


def test_field_schema_is_consistent():
    assert len(FIELDS) == len(set(FIELDS))
    assert set(AMOUNT_FIELDS) <= set(FIELDS)
    assert set(DATE_FIELDS) <= set(FIELDS)
    assert not set(AMOUNT_FIELDS) & set(DATE_FIELDS)
    assert FIELD_INDEX == {n: i for i, n in enumerate(FIELDS)}


def test_field_truth_all_spans_orders_primary_first():
    truth = FieldTruth("total", value="1", present=True, span=(3, 4), alt_spans=[(9, 9)])
    assert truth.all_spans == [(3, 4), (9, 9)]
    assert truth.to_dict()["n_occurrences"] == 2


def test_field_truth_absent_has_no_spans():
    truth = FieldTruth("tax")
    assert truth.all_spans == []
    assert truth.to_dict()["span_start"] == -1
