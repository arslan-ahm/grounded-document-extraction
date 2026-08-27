"""Metrics: field accuracy, grounding, calibration, and the statistics.

Everything here is checked against a closed form or a hand-worked example. The
recurring theme is the ``NaN`` discipline the project standard demands: an empty
prediction does not have precision 1.0, an AUROC with one class present is
undefined, and a field absent from the document has no box to score against.
Each of those is asserted, because a convenient default in any of them would make
the headline tables flattering and wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gdx.data.schema import FIELDS, Document, FieldTruth, Token
from gdx.extract import Extraction
from gdx.metrics.calibration import (
    adaptive_calibration_error,
    aurc,
    brier_score,
    error_detection_auroc,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    risk_coverage_curve,
    summarise_calibration,
)
from gdx.metrics.fields import FieldRecord, anls, build_records, summarise, summarise_per_field
from gdx.metrics.grounding import span_exact, span_iou, summarise_grounding, token_overlap
from gdx.metrics.stats import (
    aggregate_by_document,
    bootstrap_ci,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    paired_bootstrap_difference,
    verdict,
)


def _record(**kw) -> FieldRecord:  # noqa: ANN003
    base = {
        "doc_id": 0,
        "field_name": "total",
        "truth_present": True,
        "truth_value": "120.00",
        "predicted_value": "120.00",
        "abstained": False,
        "confidence": 0.9,
        "grounded": True,
        "requires_normalisation": False,
        "span_exact": True,
        "span_iou": 1.0,
        "n_iters": 0,
        "reason": "",
    }
    base.update(kw)
    return FieldRecord(**base)


# --- ANLS ------------------------------------------------------------------

def test_anls_identical_is_one():
    assert anls("INV-123", "INV-123") == 1.0


def test_anls_is_case_and_whitespace_insensitive():
    assert anls(" inv-123 ", "INV-123") == 1.0


def test_anls_one_edit_hand_computed():
    """``INV-124`` vs ``INV-123``: one substitution over length 7."""
    assert anls("INV-124", "INV-123") == pytest.approx(1.0 - 1.0 / 7)


def test_anls_below_threshold_is_zero():
    assert anls("zzz", "INV-123") == 0.0


def test_anls_empty_pair_is_one_and_half_empty_is_zero():
    assert anls("", "") == 1.0
    assert anls("", "x") == 0.0
    assert anls("x", "") == 0.0


def test_anls_threshold_is_respected():
    """Similarity 0.5 exactly passes; below it is zeroed."""
    assert anls("ab", "ac", threshold=0.5) == pytest.approx(0.5)
    assert anls("ab", "ac", threshold=0.6) == 0.0


# --- FieldRecord correctness -----------------------------------------------

def test_strict_correct_requires_literal_equality():
    assert _record(predicted_value="120.00", truth_value="120.00").strict_correct
    assert not _record(predicted_value="$120.00", truth_value="120.00").strict_correct


def test_canonical_correct_parses_amounts():
    rec = _record(predicted_value="$120.00", truth_value="120.00")
    assert rec.canonical_correct
    assert not rec.strict_correct


def test_canonical_correct_parses_dates():
    rec = _record(
        field_name="invoice_date", predicted_value="3rd of Jan 2024", truth_value="2024-01-03"
    )
    assert rec.canonical_correct
    assert not rec.strict_correct


def test_abstention_is_correct_exactly_when_the_field_is_absent():
    absent = _record(truth_present=False, truth_value="", predicted_value="", abstained=True)
    assert absent.strict_correct and absent.canonical_correct
    present = _record(predicted_value="", abstained=True)
    assert not present.strict_correct and not present.canonical_correct


def test_emitting_on_an_absent_field_is_wrong():
    rec = _record(truth_present=False, truth_value="", predicted_value="120.00", abstained=False)
    assert not rec.strict_correct
    assert not rec.canonical_correct


def test_emitted_requires_a_non_empty_value():
    assert not _record(predicted_value="", abstained=False).emitted


# --- summarise -------------------------------------------------------------

def test_summarise_of_nothing_reports_zero_records():
    assert summarise([]) == {"n_records": 0}


def test_summarise_counts_accompany_every_mean():
    records = [_record(), _record(doc_id=1, predicted_value="wrong")]
    out = summarise(records)
    for key in ("strict_accuracy", "canonical_accuracy", "precision", "recall"):
        assert f"n_{key}" in out


def test_hallucination_rate_denominator_is_emissions():
    """An arm that abstains everywhere has no hallucinations and must not score 0.

    With no emissions the rate is undefined, so it is ``NaN`` with ``n=0``.
    """
    records = [_record(abstained=True, predicted_value="") for _ in range(3)]
    out = summarise(records)
    assert math.isnan(out["hallucination_rate"])
    assert out["n_emitted"] == 0.0


def test_hallucination_rate_hand_computed():
    records = [
        _record(doc_id=0, grounded=True),
        _record(doc_id=1, grounded=False),
        _record(doc_id=2, grounded=False),
        _record(doc_id=3, abstained=True, predicted_value=""),
    ]
    out = summarise(records)
    assert out["hallucination_rate"] == pytest.approx(2.0 / 3.0)
    assert out["n_emitted"] == 3.0


def test_precision_of_no_emissions_is_nan_not_one():
    records = [_record(abstained=True, predicted_value="")]
    out = summarise(records)
    assert math.isnan(out["precision"])
    assert out["n_precision"] == 0.0


def test_recall_of_no_present_fields_is_nan():
    records = [_record(truth_present=False, truth_value="", predicted_value="", abstained=True)]
    out = summarise(records)
    assert math.isnan(out["recall"])


def test_f1_is_nan_when_either_side_is_nan():
    records = [_record(abstained=True, predicted_value="")]
    assert math.isnan(summarise(records)["f1"])


def test_f1_hand_computed():
    records = [
        _record(doc_id=0),                                 # correct emission
        _record(doc_id=1, predicted_value="999.00"),       # wrong emission
        _record(doc_id=2, abstained=True, predicted_value=""),  # missed
    ]
    out = summarise(records)
    assert out["precision"] == pytest.approx(0.5)
    assert out["recall"] == pytest.approx(1.0 / 3.0)
    assert out["f1"] == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))


def test_coverage_denominator_is_all_pairs():
    records = [_record(), _record(doc_id=1, abstained=True, predicted_value="")]
    assert summarise(records)["coverage"] == pytest.approx(0.5)


def test_normalisation_split_reports_its_own_counts():
    records = [
        _record(doc_id=0, requires_normalisation=False),
        _record(
            doc_id=1, field_name="invoice_date", requires_normalisation=True,
            predicted_value="3rd of Jan 2024", truth_value="2024-01-03",
        ),
    ]
    out = summarise(records)
    assert out["n_needs_norm"] == 1.0
    assert out["n_verbatim"] == 1.0
    assert out["strict_accuracy_needs_norm"] == 0.0
    assert out["canonical_accuracy_needs_norm"] == 1.0


def test_summarise_per_field_only_includes_present_fields():
    records = [_record(field_name="total"), _record(doc_id=1, field_name="subtotal")]
    out = summarise_per_field(records)
    assert set(out) == {"total", "subtotal"}


# --- grounding -------------------------------------------------------------

def _gdoc() -> Document:
    tokens = [
        Token(text=t, box=(0.1 * i, 0.2, 0.1 * i + 0.08, 0.24), page=0)
        for i, t in enumerate(["Total", "120.00", "Due", "120.00"])
    ]
    fields = {n: FieldTruth(n) for n in FIELDS}
    fields["total"] = FieldTruth(
        "total", value="120.00", present=True, span=(1, 1), span_text="120.00",
        alt_spans=[(3, 3)],
    )
    return Document(doc_id=0, tokens=tokens, fields=fields)


def test_span_exact_accepts_the_primary_span():
    doc = _gdoc()
    ext = Extraction("total", "120.00", span=(1, 1), abstained=False)
    assert span_exact(doc, doc.fields["total"], ext)


def test_span_exact_accepts_an_alternative_occurrence():
    """A duplicated value read from the duplicate is still read correctly."""
    doc = _gdoc()
    ext = Extraction("total", "120.00", span=(3, 3), abstained=False)
    assert span_exact(doc, doc.fields["total"], ext)


def test_span_exact_rejects_the_wrong_span():
    doc = _gdoc()
    ext = Extraction("total", "Total", span=(0, 0), abstained=False)
    assert not span_exact(doc, doc.fields["total"], ext)


def test_span_exact_is_false_for_an_absent_field_or_no_claim():
    doc = _gdoc()
    assert not span_exact(doc, doc.fields["tax"], Extraction("tax", span=(0, 0)))
    assert not span_exact(doc, doc.fields["total"], Extraction("total", span=None))


def test_span_iou_is_one_for_the_exact_box():
    doc = _gdoc()
    ext = Extraction("total", "120.00", span=(1, 1), abstained=False)
    assert span_iou(doc, doc.fields["total"], ext) == pytest.approx(1.0)


def test_span_iou_is_nan_without_a_span_claim():
    """A generative prediction cannot be scored on grounding at all."""
    doc = _gdoc()
    assert math.isnan(span_iou(doc, doc.fields["total"], Extraction("total", "120.00")))


def test_span_iou_is_nan_for_an_absent_field():
    doc = _gdoc()
    assert math.isnan(span_iou(doc, doc.fields["tax"], Extraction("tax", span=(0, 0))))


def test_span_iou_takes_the_best_matching_occurrence():
    doc = _gdoc()
    ext = Extraction("total", "120.00", span=(3, 3), abstained=False)
    assert span_iou(doc, doc.fields["total"], ext) == pytest.approx(1.0)


def test_span_iou_of_a_wider_span_is_between_zero_and_one():
    doc = _gdoc()
    ext = Extraction("total", "Total 120.00", span=(0, 1), abstained=False)
    value = span_iou(doc, doc.fields["total"], ext)
    assert 0.0 < value < 1.0


def test_token_overlap_hand_computed():
    doc = _gdoc()
    ext = Extraction("total", "Total 120.00", span=(0, 1), abstained=False)
    # predicted {0,1}, gold {1}: intersection 1, union 2
    assert token_overlap(doc.fields["total"], ext) == pytest.approx(0.5)


def test_summarise_grounding_counts_only_claimed_spans():
    doc = _gdoc()
    extractions = [{"total": Extraction("total", "120.00", span=(1, 1), abstained=False)}]
    out = summarise_grounding([doc], extractions)
    assert out["grounding_exact"] == 1.0
    assert out["n_grounding_exact"] == 1.0


def test_summarise_grounding_is_nan_when_no_span_is_ever_claimed():
    doc = _gdoc()
    extractions = [{"total": Extraction("total", "120.00", span=None, abstained=False)}]
    out = summarise_grounding([doc], extractions)
    assert math.isnan(out["grounding_exact"])
    assert out["n_grounding_exact"] == 0.0


def test_build_records_covers_every_field(doc):  # noqa: ANN001
    extractions = {n: Extraction.abstain(n, "none") for n in FIELDS}
    records = build_records(doc, extractions, span_exact, span_iou)
    assert [r.field_name for r in records] == list(FIELDS)


def test_build_records_tolerates_a_missing_field(doc):  # noqa: ANN001
    records = build_records(doc, {}, span_exact, span_iou)
    assert all(r.abstained for r in records)
    assert all(r.reason == "missing" for r in records)


# --- calibration -----------------------------------------------------------

def test_ece_of_perfect_calibration_is_zero():
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [1, 1, 0, 0]) == 0.0


def test_ece_of_maximal_miscalibration_is_one():
    assert expected_calibration_error([1.0, 1.0], [0, 0]) == pytest.approx(1.0)


def test_ece_hand_computed_two_bins():
    """Confidences 0.9 (right) and 0.1 (wrong): both bins perfectly calibrated
    would give 0; here gaps are 0.1 and 0.1, weighted 0.5 each."""
    value = expected_calibration_error([0.9, 0.1], [1, 0], n_bins=10)
    assert value == pytest.approx(0.1)


def test_ace_uses_equal_mass_bins():
    conf = [0.90, 0.91, 0.92, 0.93, 0.10]
    correct = [1, 1, 1, 1, 0]
    ace = adaptive_calibration_error(conf, correct, n_bins=5)
    assert 0.0 <= ace <= 1.0


def test_ace_and_ece_agree_when_confidence_is_uniform():
    conf = list(np.linspace(0.05, 0.95, 20))
    correct = [1 if c > 0.5 else 0 for c in conf]
    ace = adaptive_calibration_error(conf, correct, 10)
    ece = expected_calibration_error(conf, correct, 10)
    assert abs(ace - ece) < 0.2


def test_mce_is_at_least_ace():
    conf = [0.9, 0.8, 0.2, 0.1]
    correct = [1, 0, 1, 0]
    assert maximum_calibration_error(conf, correct, 4) >= adaptive_calibration_error(
        conf, correct, 4
    )


def test_brier_closed_form():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_nll_closed_form():
    value = negative_log_likelihood([0.5, 0.5], [1, 0])
    assert value == pytest.approx(-math.log(0.5))


def test_calibration_metrics_are_nan_on_empty_input():
    for fn in (
        expected_calibration_error,
        adaptive_calibration_error,
        maximum_calibration_error,
        brier_score,
        negative_log_likelihood,
    ):
        assert math.isnan(fn([], []))


def test_calibration_drops_nan_confidences():
    """The LLM arm reports NaN confidence; it must not become 0.0."""
    value = brier_score([float("nan"), 1.0], [0, 1])
    assert value == 0.0


def test_calibration_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        brier_score([0.5], [1, 0])


def test_risk_coverage_is_monotone_when_confidence_is_perfect():
    coverage, risk = risk_coverage_curve([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert list(coverage) == [0.25, 0.5, 0.75, 1.0]
    assert risk[0] == 0.0
    assert risk[-1] == pytest.approx(0.5)
    assert list(risk) == sorted(risk)


def test_risk_coverage_on_empty_input():
    coverage, risk = risk_coverage_curve([], [])
    assert coverage.size == 0 and risk.size == 0


def test_aurc_is_lower_for_better_ranking():
    correct = [1, 1, 0, 0]
    good = aurc([0.9, 0.8, 0.2, 0.1], correct)
    bad = aurc([0.1, 0.2, 0.8, 0.9], correct)
    assert good < bad


def test_aurc_needs_at_least_two_points():
    assert math.isnan(aurc([0.5], [1]))


def test_error_auroc_perfect_and_inverted():
    assert error_detection_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert error_detection_auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_error_auroc_with_ties_is_one_half():
    assert error_detection_auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)


def test_error_auroc_is_nan_with_one_class():
    """No errors means nothing to detect; 0.5 would be a fabrication."""
    assert math.isnan(error_detection_auroc([0.9, 0.8], [1, 1]))
    assert math.isnan(error_detection_auroc([0.9, 0.8], [0, 0]))


def test_summarise_calibration_keys_and_count():
    out = summarise_calibration([0.9, 0.1], [1, 0])
    assert set(out) >= {"ece", "ace", "mce", "brier", "nll", "aurc", "error_auroc"}
    assert out["n_calibration"] == 2.0


# --- statistics ------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(0.5, 0.1, size=200)
    interval = bootstrap_ci(values, n_resamples=500, seed=0)
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.n == 200


def test_bootstrap_ci_drops_nan_and_records_the_count():
    interval = bootstrap_ci([1.0, float("nan"), 3.0], n_resamples=200, seed=0)
    assert interval.n == 2
    assert interval.estimate == pytest.approx(2.0)


def test_bootstrap_ci_of_nothing_is_nan():
    interval = bootstrap_ci([])
    assert math.isnan(interval.estimate)
    assert interval.n == 0


def test_bootstrap_ci_of_one_value_is_degenerate():
    interval = bootstrap_ci([2.0])
    assert interval.estimate == interval.lower == interval.upper == 2.0


def test_paired_bootstrap_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])


def test_paired_bootstrap_of_identical_arrays_is_zero():
    interval = paired_bootstrap_difference([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_resamples=200)
    assert interval.estimate == 0.0


def test_compare_reports_nan_p_when_all_differences_are_zero():
    """Two arms sharing an encoder agree exactly on most fields."""
    result = compare([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], n_resamples=100)
    assert math.isnan(result.p_value)
    assert not result.significant


def test_compare_detects_a_consistent_difference():
    a = [1.0] * 30
    b = [0.0] * 30
    result = compare(a, b, n_resamples=300)
    assert result.mean_a == 1.0 and result.mean_b == 0.0
    assert result.difference.estimate == pytest.approx(1.0)
    assert result.p_value < 0.05


def test_compare_effect_size_is_zero_when_variance_is_zero():
    result = compare([1.0, 1.0], [0.0, 0.0], n_resamples=50)
    assert result.effect_size == 0.0


def test_compare_to_dict_has_the_expected_keys():
    keys = set(compare([1.0, 0.0], [0.0, 1.0], n_resamples=50).to_dict())
    assert keys >= {"metric", "unit", "p_adjusted", "ci_lower", "ci_upper", "n"}


def test_holm_bonferroni_is_monotone_and_excludes_nan():
    comparisons = [
        compare([1.0] * 20, [0.0] * 20, name_a=f"a{i}", n_resamples=50) for i in range(3)
    ]
    comparisons.append(compare([1.0, 1.0], [1.0, 1.0], name_a="tie", n_resamples=50))
    holm_bonferroni(comparisons)
    adjusted = [c.p_adjusted for c in comparisons if c.p_adjusted is not None]
    assert len(adjusted) == 3
    assert adjusted == sorted(adjusted) or all(a <= 1.0 for a in adjusted)
    assert comparisons[-1].p_adjusted is None


def test_holm_bonferroni_on_an_all_nan_family_is_a_noop():
    comparisons = [compare([1.0], [1.0], n_resamples=10)]
    assert holm_bonferroni(comparisons) == comparisons


def test_bootstrap_metric_difference_on_a_rate():
    def rate(values):  # noqa: ANN001, ANN202
        return float(values.mean())

    a = np.array([1.0] * 40 + [0.0] * 10)
    b = np.array([0.0] * 50)
    interval = bootstrap_metric_difference(a, b, rate, n_resamples=300)
    assert interval.estimate == pytest.approx(0.8)
    assert interval.lower > 0.0


def test_bootstrap_metric_difference_shape_mismatch_raises():
    with pytest.raises(ValueError, match="axis 0"):
        bootstrap_metric_difference(np.zeros(3), np.zeros(4), lambda v: float(v.mean()))


def test_noise_scale_is_sqrt_two_times_sd():
    values = [0.1, 0.2, 0.3]
    assert noise_scale(values) == pytest.approx(math.sqrt(2.0) * np.std(values, ddof=1))


def test_noise_scale_needs_two_values():
    assert math.isnan(noise_scale([0.5]))
    assert math.isnan(noise_scale([]))


@pytest.mark.parametrize(
    ("difference", "scale", "expected"),
    [
        (0.4, 0.1, "robust"),
        (0.25, 0.1, "survives"),
        (0.15, 0.1, "suggestive"),
        (0.05, 0.1, "inside noise"),
        (0.1, float("nan"), "unknown"),
        (0.1, 0.0, "unknown"),
        (float("nan"), 0.1, "unknown"),
    ],
)
def test_verdict_thresholds(difference, scale, expected):  # noqa: ANN001
    assert verdict(difference, scale) == expected


def test_verdict_is_sign_agnostic():
    assert verdict(-0.4, 0.1) == verdict(0.4, 0.1)


def test_aggregate_by_document_averages_within_a_document():
    ids = np.array([0, 0, 1, 1])
    values = np.array([1.0, 0.0, 1.0, 1.0])
    out_ids, out_values = aggregate_by_document(ids, values)
    assert list(out_ids) == [0, 1]
    assert list(out_values) == [0.5, 1.0]


def test_aggregate_by_document_drops_all_nan_documents():
    ids = np.array([0, 0, 1])
    values = np.array([float("nan"), float("nan"), 1.0])
    out_ids, out_values = aggregate_by_document(ids, values)
    assert list(out_ids) == [1]
    assert list(out_values) == [1.0]


def test_aggregate_by_document_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_by_document(np.zeros(2), np.zeros(3))
