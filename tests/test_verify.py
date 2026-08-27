"""The bounded verification loop: each check, the repair, and the bound.

The loop is the mechanism this repository adds on top of selection, so every
branch is exercised: each check in isolation, the re-selection path, the
abstention path, the ``abstain=False`` variant that emits anyway, and the
iteration bound. The arithmetic check is tested against closed forms, including
the case where it *does not apply* -- returning 0.0 there instead of ``None``
would inflate the reported consistency rate on every document with a dropped
subtotal.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from gdx.config import VerifyConfig
from gdx.data.schema import AMOUNT_FIELDS, FIELDS, Document, FieldTruth, Token
from gdx.extract import Candidate
from gdx.verify import (
    VerificationTrace,
    arithmetic_ok,
    arithmetic_residual,
    provenance_ok,
    verify_document,
)

BASE = VerifyConfig()


def _doc(texts: list[str], pages: list[int] | None = None) -> Document:
    pages = pages or [0] * len(texts)
    tokens = [
        Token(text=t, box=(0.05 * i, 0.1, 0.05 * i + 0.04, 0.13), page=p)
        for i, (t, p) in enumerate(zip(texts, pages, strict=True))
    ]
    return Document(doc_id=0, tokens=tokens, fields={n: FieldTruth(n) for n in FIELDS})


def _only(field_name: str, candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    out = {n: [] for n in FIELDS}
    out[field_name] = candidates
    return out


# --- arithmetic ------------------------------------------------------------

def test_arithmetic_residual_closed_form():
    values = {"subtotal": "100.00", "tax": "20.00", "total": "120.00"}
    assert arithmetic_residual(values) == pytest.approx(0.0)
    values["total"] = "125.00"
    assert arithmetic_residual(values) == pytest.approx(-5.0)


def test_arithmetic_residual_handles_currency_styles():
    values = {"subtotal": "$1,000.00", "tax": "EUR 200", "total": "1200.00"}
    assert arithmetic_residual(values) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "values",
    [
        {"subtotal": "", "tax": "20.00", "total": "120.00"},
        {"subtotal": "100.00", "tax": "", "total": "120.00"},
        {"subtotal": "100.00", "tax": "20.00", "total": ""},
        {"subtotal": "Total", "tax": "20.00", "total": "120.00"},
        {},
    ],
)
def test_arithmetic_residual_is_nan_when_it_cannot_apply(values):  # noqa: ANN001
    assert math.isnan(arithmetic_residual(values))


def test_arithmetic_ok_returns_none_when_it_cannot_apply():
    """``None``, not ``False``: an inapplicable check is not a failed one."""
    assert arithmetic_ok({"subtotal": "", "tax": "1", "total": "1"}, 0.01) is None


def test_arithmetic_ok_respects_the_tolerance():
    values = {"subtotal": "100.00", "tax": "20.00", "total": "120.01"}
    assert arithmetic_ok(values, 0.011) is True
    assert arithmetic_ok(values, 0.005) is False


def test_arithmetic_ok_on_exact_agreement():
    assert arithmetic_ok({"subtotal": "1", "tax": "2", "total": "3"}, 0.0) is True


# --- provenance ------------------------------------------------------------

def test_provenance_ok_accepts_a_document_substring():
    doc = _doc(["Total", "Due", "$", "120.00"])
    assert provenance_ok(doc, "total", "$ 120.00")
    assert provenance_ok(doc, "total", "120.00")


def test_provenance_ok_rejects_an_absent_value():
    doc = _doc(["Total", "Due", "$", "120.00"])
    assert not provenance_ok(doc, "total", "999.99")
    assert not provenance_ok(doc, "total", "")


def test_provenance_ok_rejects_a_cross_page_splice():
    doc = _doc(["120.00", "Page", "of", "2", "Invoice"], pages=[0, 0, 1, 1, 1])
    assert not provenance_ok(doc, "vendor_name", "Page of")


# --- stage 1: per-field checks --------------------------------------------

def test_verification_disabled_emits_the_top_candidate_unchecked():
    doc = _doc(["Total", "999.99"])
    cands = _only("invoice_id", [Candidate(value="NOT-IN-DOC", prob=0.9, span=None)])
    out, trace = verify_document(doc, cands, replace(BASE, enabled=False))
    assert out["invoice_id"].value == "NOT-IN-DOC"
    assert out["invoice_id"].grounded is False
    assert trace.n_iters == 0


def test_type_check_rejects_a_wrong_typed_span():
    doc = _doc(["Bracket", "INV-12345"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.9, span=(0, 0))])
    out, trace = verify_document(doc, cands, BASE)
    assert out["invoice_id"].abstained
    assert trace.type_rejections == 1


def test_type_check_off_lets_the_wrong_type_through():
    doc = _doc(["Bracket", "INV-12345"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.9, span=(0, 0))])
    out, _ = verify_document(doc, cands, replace(BASE, type_check=False))
    assert out["invoice_id"].value == "Bracket"
    assert out["invoice_id"].grounded is True


def test_reselection_takes_the_next_admissible_candidate():
    doc = _doc(["Bracket", "INV-12345"])
    cands = _only(
        "invoice_id",
        [
            Candidate(value="", prob=0.6, span=(0, 0)),  # wrong type
            Candidate(value="", prob=0.3, span=(1, 1)),  # correct
        ],
    )
    out, trace = verify_document(doc, cands, BASE)
    assert out["invoice_id"].value == "INV-12345"
    assert out["invoice_id"].span == (1, 1)
    assert trace.reselections == 1
    assert trace.type_rejections == 1


def test_reselection_is_bounded_by_max_iters():
    doc = _doc(["a", "b", "c", "d", "INV-12345"])
    cands = _only(
        "invoice_id",
        [Candidate(value="", prob=0.9 - 0.1 * i, span=(i, i)) for i in range(5)],
    )
    out, trace = verify_document(doc, cands, replace(BASE, max_iters=2))
    assert out["invoice_id"].abstained
    assert trace.n_iters <= 2
    out2, _ = verify_document(doc, cands, replace(BASE, max_iters=5))
    assert out2["invoice_id"].value == "INV-12345"


def test_abstain_off_emits_the_failing_candidate_with_a_reason():
    doc = _doc(["Bracket"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.9, span=(0, 0))])
    out, _ = verify_document(doc, cands, replace(BASE, abstain=False))
    assert not out["invoice_id"].abstained
    assert out["invoice_id"].value == "Bracket"
    assert out["invoice_id"].reason == "checks_failed_emitted_anyway"


def test_no_candidates_means_abstain_with_that_reason():
    doc = _doc(["INV-12345"])
    out, _ = verify_document(doc, {n: [] for n in FIELDS}, BASE)
    assert all(e.abstained for e in out.values())
    assert out["invoice_id"].reason == "no_candidate"


def test_confidence_threshold_abstains_on_low_confidence():
    doc = _doc(["INV-12345"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.2, span=(0, 0))])
    out, _ = verify_document(doc, cands, replace(BASE, min_confidence=0.5))
    assert out["invoice_id"].abstained
    assert out["invoice_id"].reason == "low_confidence"


def test_confidence_threshold_keeps_high_confidence():
    doc = _doc(["INV-12345"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.8, span=(0, 0))])
    out, _ = verify_document(doc, cands, replace(BASE, min_confidence=0.5))
    assert out["invoice_id"].value == "INV-12345"


def test_checks_dict_records_what_ran():
    doc = _doc(["INV-12345"])
    cands = _only("invoice_id", [Candidate(value="", prob=0.9, span=(0, 0))])
    out, _ = verify_document(doc, cands, BASE)
    assert out["invoice_id"].checks["type"] is True
    assert out["invoice_id"].checks["provenance"] is True


# --- stage 2: arithmetic repair -------------------------------------------

def _amount_doc() -> Document:
    return _doc(["100.00", "20.00", "120.00", "999.00"])


def _amount_cands(subtotal, tax, total):  # noqa: ANN001, ANN202
    out = {n: [] for n in FIELDS}
    for name, spans in (("subtotal", subtotal), ("tax", tax), ("total", total)):
        out[name] = [Candidate(value="", prob=0.9 - 0.1 * i, span=s) for i, s in enumerate(spans)]
    return out


def test_arithmetic_check_passes_on_consistent_amounts():
    doc = _amount_doc()
    out, trace = verify_document(doc, _amount_cands([(0, 0)], [(1, 1)], [(2, 2)]), BASE)
    assert trace.arithmetic_checked
    assert trace.arithmetic_final_ok is True
    assert out["total"].value == "120.00"
    assert all(out[n].checks["arithmetic"] is True for n in AMOUNT_FIELDS)


def test_arithmetic_repair_reselects_to_restore_consistency():
    """The wrong total is chosen first; the loop must fall back to the right one."""
    doc = _amount_doc()
    cands = _amount_cands([(0, 0)], [(1, 1)], [(3, 3), (2, 2)])
    out, trace = verify_document(doc, cands, BASE)
    assert out["total"].value == "120.00"
    assert trace.arithmetic_rejections >= 1
    assert trace.reselections >= 1
    assert trace.arithmetic_final_ok is True


def test_arithmetic_failure_with_no_alternative_abstains_on_all_three():
    doc = _amount_doc()
    cands = _amount_cands([(0, 0)], [(1, 1)], [(3, 3)])
    out, trace = verify_document(doc, cands, BASE)
    assert all(out[n].abstained for n in AMOUNT_FIELDS)
    assert all(out[n].reason == "arithmetic_failed" for n in AMOUNT_FIELDS)
    assert trace.arithmetic_final_ok is False


def test_arithmetic_failure_with_abstain_off_emits_and_flags():
    doc = _amount_doc()
    cands = _amount_cands([(0, 0)], [(1, 1)], [(3, 3)])
    out, _ = verify_document(doc, cands, replace(BASE, abstain=False))
    assert not out["total"].abstained
    assert out["total"].reason == "arithmetic_failed_emitted_anyway"
    assert out["total"].checks["arithmetic"] is False


def test_arithmetic_off_skips_the_repair_entirely():
    doc = _amount_doc()
    cands = _amount_cands([(0, 0)], [(1, 1)], [(3, 3)])
    out, trace = verify_document(doc, cands, replace(BASE, arithmetic=False))
    assert out["total"].value == "999.00"
    assert trace.arithmetic_checked is False
    assert trace.arithmetic_final_ok is None


def test_arithmetic_does_not_apply_when_a_field_is_abstained():
    doc = _amount_doc()
    cands = _amount_cands([], [(1, 1)], [(2, 2)])
    out, trace = verify_document(doc, cands, BASE)
    assert trace.arithmetic_checked is False
    assert out["subtotal"].abstained
    assert out["total"].checks["arithmetic"] is None


def test_arithmetic_repair_is_bounded():
    """With max_iters=1 the loop cannot chase more than one substitution."""
    doc = _doc(["100.00", "20.00", "999.00", "888.00", "120.00"])
    cands = _amount_cands([(0, 0)], [(1, 1)], [(2, 2), (3, 3), (4, 4)])
    out, trace = verify_document(doc, cands, replace(BASE, max_iters=1))
    assert trace.arithmetic_rejections <= 1
    assert out["total"].abstained or out["total"].value in {"888.00", "120.00"}


# --- trace -----------------------------------------------------------------

def test_trace_to_dict_keys():
    trace = VerificationTrace()
    keys = set(trace.to_dict())
    assert "verify_iters" in keys
    assert "verify_arithmetic_ok" in keys
    assert trace.to_dict()["verify_arithmetic_ok"] is None


def test_verify_returns_every_field_key():
    doc = _doc(["INV-12345"])
    out, _ = verify_document(doc, {n: [] for n in FIELDS}, BASE)
    assert set(out) == set(FIELDS)


def test_verify_on_an_empty_document_abstains_everywhere():
    doc = _doc([])
    cands = {n: [Candidate(value="", prob=0.9, span=(0, 0))] for n in FIELDS}
    out, _ = verify_document(doc, cands, BASE)
    assert all(e.abstained for e in out.values())


def test_verify_is_deterministic():
    doc = _amount_doc()
    cands = _amount_cands([(0, 0)], [(1, 1)], [(3, 3), (2, 2)])
    a, ta = verify_document(doc, cands, BASE)
    b, tb = verify_document(doc, cands, BASE)
    assert {k: v.to_dict() for k, v in a.items()} == {k: v.to_dict() for k, v in b.items()}
    assert ta.to_dict() == tb.to_dict()
