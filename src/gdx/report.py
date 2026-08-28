"""Render the committed CSVs as Markdown tables.

The documentation quotes numbers. Hand-typing them is how a wrong table gets
shipped, so **every table in ``README.md`` and ``docs/*.md`` is produced by this
module from the CSV it belongs to**, injected between markers by
``scripts/render_docs.py``. ``--check`` fails if a document has drifted, and
``tests/test_render_docs.py`` runs that check, so a stale number is a test
failure rather than something a reader has to notice.

If a CSV is missing, the renderer emits ``_not measured_`` rather than inventing
a row. That is the only honest placeholder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("results/tables")


def _fmt(value: object, digits: int = 4) -> str:
    """Format one cell, keeping ``NaN`` visible as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if number == int(number) and abs(number) < 1e6 and digits == 0:
        return str(int(number))
    if abs(number) >= 1e5 or (abs(number) < 1e-3 and number != 0):
        return f"{number:.2e}"
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def to_markdown(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    digits: int = 4,
    bold_max: str | None = None,
    bold_min: str | None = None,
    rename: dict[str, str] | None = None,
) -> str:
    """Render a frame as a GitHub Markdown table.

    Args:
        frame: The data.
        columns: Column subset, in order. Missing columns are skipped silently so
            a renderer keeps working when an optional metric is absent.
        digits: Decimal places.
        bold_max: Column whose maximum is bolded.
        bold_min: Column whose minimum is bolded.
        rename: Header relabelling.
    """
    if frame is None or frame.empty:
        return "_not measured_"
    cols = [c for c in (columns or list(frame.columns)) if c in frame.columns]
    if not cols:
        return "_not measured_"
    view = frame[cols]
    headers = [(rename or {}).get(c, c) for c in cols]
    best: dict[str, float] = {}
    for name, reducer in ((bold_max, np.nanmax), (bold_min, np.nanmin)):
        if name and name in view.columns:
            values = pd.to_numeric(view[name], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(values).any():
                best[name] = float(reducer(values))

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        cells = []
        for col in cols:
            text = _fmt(row[col], digits)
            if col in best:
                try:
                    if np.isfinite(float(row[col])) and float(row[col]) == best[col]:
                        text = f"**{text}**"
                except (TypeError, ValueError):
                    pass
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _read(name: str) -> pd.DataFrame | None:
    path = TABLES / f"{name}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def table_method() -> str:
    """The headline comparison of all seven arms."""
    return to_markdown(
        _read("method_comparison"),
        [
            "arm", "family", "strict_accuracy", "canonical_accuracy", "coverage",
            "hallucination_rate", "grounding_exact", "grounding_iou", "f1",
            "absent_abstain_rate", "n_records",
        ],
        bold_max="canonical_accuracy",
    )


def table_verdicts() -> str:
    """Every arm against the reference approach, placed on the noise scale."""
    return to_markdown(
        _read("verdicts"),
        ["arm", "metric", "value", "reference_value", "delta", "noise_scale",
         "ratio_to_noise", "verdict"],
    )


def table_verdicts_vs_heuristic() -> str:
    """Every arm against the rule baseline, the stronger of the two references."""
    return to_markdown(
        _read("verdicts_vs_heuristic"),
        ["arm", "metric", "value", "reference_value", "delta", "noise_scale",
         "ratio_to_noise", "verdict"],
    )


def table_verdicts_vs_span_only() -> str:
    """Every arm against selection with verification off, isolating the loop."""
    return to_markdown(
        _read("verdicts_vs_span_only"),
        ["arm", "metric", "value", "reference_value", "delta", "noise_scale",
         "ratio_to_noise", "verdict"],
    )


def table_seed_variance() -> str:
    return to_markdown(
        _read("seed_variance"),
        ["arm", "metric", "n_seeds", "mean", "sd", "noise_scale", "min", "max"],
    )


def table_statistical_tests() -> str:
    return to_markdown(
        _read("statistical_tests"),
        ["name_a", "name_b", "metric", "unit", "mean_a", "mean_b", "difference",
         "ci_lower", "ci_upper", "p_value", "p_adjusted", "effect_size", "n",
         "significant"],
    )


def table_normalisation_cost() -> str:
    """Accuracy split by whether the written form requires normalisation."""
    return to_markdown(
        _read("normalisation_cost"),
        ["arm", "subset", "n", "strict_accuracy", "canonical_accuracy", "coverage"],
    )


def table_per_field() -> str:
    return to_markdown(
        _read("per_field"),
        ["arm", "field", "n", "n_present", "strict_accuracy", "canonical_accuracy",
         "coverage", "grounding_exact"],
    )


def table_calibration() -> str:
    return to_markdown(
        _read("calibration"),
        ["arm", "ece", "ace", "mce", "brier", "nll", "aurc", "error_auroc",
         "mean_confidence", "n_calibration"],
    )


def table_abstention() -> str:
    return to_markdown(
        _read("abstention"),
        ["arm", "coverage", "absent_abstain_rate", "n_absent", "present_emit_rate",
         "n_present", "top_abstain_reason", "top_abstain_count", "mean_verify_iters"],
    )


def table_ablations() -> str:
    return to_markdown(
        _read("ablations"),
        ["kind", "ablation", "switch", "metric", "value", "full_value", "delta",
         "noise_scale", "ratio_to_noise", "verdict", "n_seeds"],
    )


def table_efficiency() -> str:
    return to_markdown(
        _read("efficiency"),
        ["arm", "batch_size", "seq_len", "params", "mmacs", "latency_ms", "iqr_ms",
         "latency_per_doc_ms", "macs_per_ms", "latency_reduction_vs_generative",
         "params_reduction_vs_generative", "macs_reduction_vs_generative"],
    )


def table_decode_scaling() -> str:
    return to_markdown(
        _read("decode_scaling"),
        ["head", "dec_max_len", "latency_ms", "iqr_ms", "fitted_exponent"],
    )


def table_baseline_cost() -> str:
    return to_markdown(_read("baseline_cost"), ["arm", "metric", "value", "n_docs", "note"])


def table_invariant() -> str:
    """The no-hallucination guarantee, as counts with explicit denominators."""
    return to_markdown(
        _read("invariant"),
        ["source", "population", "n_seeds", "n_documents", "n_opportunities",
         "n_ungrounded", "rate"],
        digits=6,
    )


def table_generative_budget() -> str:
    """The reference approach at the shared budget and at a longer schedule."""
    return to_markdown(
        _read("generative_budget"),
        ["arm", "epochs", "strict_accuracy", "canonical_accuracy", "coverage",
         "hallucination_rate", "final_val_loss", "n_records"],
    )


def table_determinism() -> str:
    return to_markdown(_read("determinism"), ["quantity", "n", "max_abs_difference", "identical"])


def table_heuristic_sweep() -> str:
    return to_markdown(
        _read("heuristic_sweep"),
        ["variant", "split", "canonical_accuracy", "strict_accuracy", "coverage", "n_records"],
    )


#: Table name to renderer, keyed by the marker used in the documentation.
ALL: dict[str, Callable[[], str]] = {
    "method": table_method,
    "verdicts": table_verdicts,
    "verdicts_vs_heuristic": table_verdicts_vs_heuristic,
    "verdicts_vs_span_only": table_verdicts_vs_span_only,
    "seed_variance": table_seed_variance,
    "statistical_tests": table_statistical_tests,
    "normalisation_cost": table_normalisation_cost,
    "per_field": table_per_field,
    "calibration": table_calibration,
    "abstention": table_abstention,
    "ablations": table_ablations,
    "efficiency": table_efficiency,
    "decode_scaling": table_decode_scaling,
    "baseline_cost": table_baseline_cost,
    "invariant": table_invariant,
    "generative_budget": table_generative_budget,
    "determinism": table_determinism,
    "heuristic_sweep": table_heuristic_sweep,
}
