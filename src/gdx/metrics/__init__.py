"""Task metrics, grounding metrics, calibration, and statistical comparison."""

from __future__ import annotations

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
from gdx.metrics.fields import (
    FieldRecord,
    anls,
    build_records,
    summarise,
    summarise_per_field,
)
from gdx.metrics.grounding import (
    span_exact,
    span_iou,
    summarise_grounding,
    token_overlap,
)
from gdx.metrics.stats import (
    Comparison,
    Interval,
    aggregate_by_document,
    bootstrap_ci,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    paired_bootstrap_difference,
    verdict,
)

__all__ = [
    "Comparison",
    "FieldRecord",
    "Interval",
    "adaptive_calibration_error",
    "aggregate_by_document",
    "anls",
    "aurc",
    "bootstrap_ci",
    "bootstrap_metric_difference",
    "brier_score",
    "build_records",
    "compare",
    "error_detection_auroc",
    "expected_calibration_error",
    "holm_bonferroni",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "noise_scale",
    "paired_bootstrap_difference",
    "risk_coverage_curve",
    "span_exact",
    "span_iou",
    "summarise",
    "summarise_calibration",
    "summarise_grounding",
    "summarise_per_field",
    "token_overlap",
    "verdict",
]
