"""Derived tables, computed from the committed per-item CSVs.

Nothing here runs a model. Every table is a function of
``results/runs/seed*/per_item.csv`` and ``results/tables/seed_runs.csv``, which
means each one is reproducible from the committed evidence alone and a reader can
recompute any cell without an hour of training.

That property is also what makes the documentation checkable: ``gdx.report``
renders these CSVs into the Markdown tables, and ``scripts/render_docs.py
--check`` fails if a document has drifted from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gdx.data.schema import FIELDS
from gdx.metrics.calibration import summarise_calibration
from gdx.pipelines.experiments import (
    METRIC_FAMILY,
    REFERENCE_ARM,
    RUNS,
    TABLES,
    paired_tests,
    seed_variance,
    verdict_table,
)
from gdx.utils.logging import get_logger

LOG = get_logger(__name__)

#: Columns shown in the headline method table, in order.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "arm",
    "family",
    "strict_accuracy",
    "canonical_accuracy",
    "coverage",
    "hallucination_rate",
    "grounding_exact",
    "grounding_iou",
    "precision",
    "recall",
    "f1",
    "anls",
    "absent_abstain_rate",
    "error_auroc",
    "aurc",
    "ece",
    "n_records",
)


def load_per_item(runs_dir: Path = RUNS) -> pd.DataFrame:
    """Concatenate every seed's per-item CSV.

    Raises:
        FileNotFoundError: If no per-item CSV exists, rather than returning an
            empty frame that would silently produce empty tables -- the exact
            failure the project standard calls out.
    """
    paths = sorted(runs_dir.glob("seed*/per_item.csv"))
    if not paths:
        raise FileNotFoundError(f"no per-item CSVs under {runs_dir}; run the experiments first")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = int(path.parent.name.replace("seed", ""))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def headline_table(runs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The main comparison, on one seed, with the columns a reader needs first."""
    frame = runs[runs["seed"] == seed].copy()
    columns = [c for c in HEADLINE_COLUMNS if c in frame.columns]
    order = {
        "heuristic": 0, "llm_stub": 1, "generative": 2, "generative_verify": 3,
        "span_only": 4, "span_verify": 5, "span_verify_norm": 6,
    }
    frame["_order"] = frame["arm"].map(order).fillna(99)
    return frame.sort_values("_order")[columns].reset_index(drop=True)


def normalisation_cost(per_item: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Accuracy split by whether the written form requires normalisation.

    This is the table that states the structural limitation of selection-only
    extraction as a number. On fields written verbatim, selection and generation
    are competing on equal terms. On fields whose target differs from the written
    form -- a date printed "3rd of Jan 2024" against a target of ``2024-01-03`` --
    selection's strict accuracy is bounded at zero by construction, and this table
    reports both the bound and how many fields it applies to.
    """
    frame = per_item[per_item["seed"] == seed]
    present = frame[frame["truth_present"]]
    rows: list[dict[str, Any]] = []
    for arm, group in present.groupby("arm"):
        for label, subset in (
            ("verbatim", group[~group["requires_normalisation"]]),
            ("needs_normalisation", group[group["requires_normalisation"]]),
        ):
            if subset.empty:
                continue
            rows.append(
                {
                    "arm": arm,
                    "subset": label,
                    "n": int(len(subset)),
                    "strict_accuracy": float(subset["strict_correct"].mean()),
                    "canonical_accuracy": float(subset["canonical_correct"].mean()),
                    "coverage": float(subset["emitted"].mean()),
                    "anls_proxy_exact": float(subset["strict_correct"].mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["subset", "arm"]).reset_index(drop=True)


def per_field_table(per_item: pd.DataFrame, seed: int, arms: tuple[str, ...] | None = None
                    ) -> pd.DataFrame:
    """Canonical accuracy and coverage per field, for the named arms."""
    frame = per_item[per_item["seed"] == seed]
    if arms:
        frame = frame[frame["arm"].isin(arms)]
    rows: list[dict[str, Any]] = []
    for (arm, field_name), group in frame.groupby(["arm", "field"]):
        present = group[group["truth_present"]]
        rows.append(
            {
                "arm": arm,
                "field": field_name,
                "n": int(len(group)),
                "n_present": int(len(present)),
                "strict_accuracy": float(group["strict_correct"].mean()),
                "canonical_accuracy": float(group["canonical_correct"].mean()),
                "coverage": float(group["emitted"].mean()),
                "grounding_exact": (
                    float(present["span_exact"].mean()) if len(present) else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["_order"] = out["field"].map({f: i for i, f in enumerate(FIELDS)})
    return out.sort_values(["arm", "_order"]).drop(columns="_order").reset_index(drop=True)


def calibration_table(per_item: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Calibration and selective-prediction metrics on emitted values only."""
    frame = per_item[(per_item["seed"] == seed) & per_item["emitted"]]
    rows: list[dict[str, Any]] = []
    for arm, group in frame.groupby("arm"):
        stats = summarise_calibration(
            group["confidence"].to_numpy(dtype=np.float64),
            group["canonical_correct"].to_numpy(dtype=np.float64),
        )
        stats["arm"] = arm
        rows.append(stats)
    out = pd.DataFrame(rows)
    return out[["arm", *[c for c in out.columns if c != "arm"]]]


def abstention_table(per_item: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Abstention behaviour: when it is right to decline, and does the arm.

    ``absent_abstain_rate`` is the one that matters: on fields the generator
    genuinely omitted, abstaining is the *correct* output and emitting anything is
    wrong. An arm that never abstains scores 0 here however accurate it looks
    elsewhere.
    """
    frame = per_item[per_item["seed"] == seed]
    rows: list[dict[str, Any]] = []
    for arm, group in frame.groupby("arm"):
        absent = group[~group["truth_present"]]
        present = group[group["truth_present"]]
        reasons = group[group["abstained"]]["reason"].value_counts()
        rows.append(
            {
                "arm": arm,
                "coverage": float(group["emitted"].mean()),
                "absent_abstain_rate": float(absent["abstained"].mean()) if len(absent) else np.nan,
                "n_absent": int(len(absent)),
                "present_emit_rate": (
                    float(present["emitted"].mean()) if len(present) else np.nan
                ),
                "n_present": int(len(present)),
                "top_abstain_reason": reasons.index[0] if len(reasons) else "",
                "top_abstain_count": int(reasons.iloc[0]) if len(reasons) else 0,
                "mean_verify_iters": float(group["n_iters"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("arm").reset_index(drop=True)


def write_all(seeds: list[int], tables_dir: Path = TABLES) -> dict[str, Path]:
    """Recompute every derived table from the committed artefacts.

    Args:
        seeds: Seeds to include. The first is the *primary* seed used for the
            single-model tables; the whole list drives the seed study.
        tables_dir: Output directory.

    Returns:
        Table name to path.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(tables_dir / "seed_runs.csv")
    runs = runs[runs["seed"].isin(seeds)]
    per_item = load_per_item()
    per_item = per_item[per_item["seed"].isin(seeds)]
    primary = seeds[0]

    written: dict[str, Path] = {}

    def dump(name: str, frame: pd.DataFrame) -> None:
        path = tables_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
        LOG.info("wrote %s (%d rows)", path, len(frame))

    dump("method_comparison", headline_table(runs, primary))
    variance = seed_variance(runs)
    dump("seed_variance", variance)
    dump("verdicts", verdict_table(runs, variance))
    dump("statistical_tests", paired_tests(per_item, primary))
    dump("normalisation_cost", normalisation_cost(per_item, primary))
    dump(
        "per_field",
        per_field_table(per_item, primary, ("generative", "span_verify", "heuristic")),
    )
    dump("calibration", calibration_table(per_item, primary))
    dump("abstention", abstention_table(per_item, primary))
    dump("seed_runs_family", runs[["arm", "seed", "family", *[
        m for m in METRIC_FAMILY if m in runs.columns
    ], "hallucination_rate"]])
    LOG.info("reference arm for verdicts: %s; primary seed: %d", REFERENCE_ARM, primary)
    return written
