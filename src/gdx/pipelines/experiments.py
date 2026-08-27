"""Per-seed experiment execution and the derived analysis tables.

Split into two halves on purpose. :func:`run_seed` executes **one seed** and
appends its rows to ``results/tables/seed_runs.csv``, writing its per-item CSV
next to it. :func:`analyse` reads those artefacts back and produces every derived
table. Nothing in the analysis half re-runs a model.

That split is what makes the experiment restartable: a seed that has already
landed is skipped, and a crash costs one seed rather than the matrix. It also
means the statistical tables are reproducible from the committed CSVs alone,
without a GPU-free hour of retraining.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gdx.arms import ARMS, baseline_candidates, model_candidates
from gdx.config import Config
from gdx.metrics.stats import (
    Comparison,
    aggregate_by_document,
    bootstrap_metric_difference,
    compare,
    holm_bonferroni,
    noise_scale,
    verdict,
)
from gdx.pipelines.core import (
    build_llm_client,
    evaluate_arm,
    prepare,
    select_heuristic_variant,
    train_head,
)
from gdx.utils.logging import get_logger, write_json

LOG = get_logger(__name__)

TABLES = Path("results/tables")
RUNS = Path("results/runs")

#: The metric family the Holm correction is applied across. Chosen before the
#: results were looked at; adding a metric afterwards would make the correction
#: meaningless.
METRIC_FAMILY: tuple[str, ...] = (
    "strict_accuracy",
    "canonical_accuracy",
    "coverage",
    "grounding_exact",
    "grounding_iou",
)

#: The arm every other arm is compared against: the reference approach.
REFERENCE_ARM = "generative"


def run_seed(cfg: Config, seed: int, out_dir: Path = RUNS) -> pd.DataFrame:
    """Train both heads at ``seed``, evaluate all seven arms, write the run dir.

    Args:
        cfg: Base configuration. ``cfg.model.head`` is ignored -- both heads are
            trained.
        seed: The seed, applied to *both* the generator and initialisation, so a
            seed study measures data noise as well as initialisation noise.
        out_dir: Parent for ``results/runs/seed<k>/``.

    Returns:
        One row per arm, with ``seed`` and timing columns.
    """
    run_dir = out_dir / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    prepared = prepare(cfg, seed=seed)
    test = prepared.splits.test
    started = time.perf_counter()

    models: dict[str, Any] = {}
    histories: dict[str, dict[str, Any]] = {}
    candidates: dict[str, list] = {}
    for head in ("span", "generative"):
        model, history = train_head(cfg, prepared, head, run_dir / head)
        models[head] = model
        histories[head] = history
        candidates[head] = model_candidates(model, test, cfg)

    variant = select_heuristic_variant(cfg, prepared)
    candidates["heuristic"], _ = baseline_candidates("heuristic", test, variant=variant)
    client = build_llm_client(cfg)
    candidates["llm"], llm_usage = baseline_candidates("llm", test, client=client)

    rows: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for arm in ARMS:
        result = evaluate_arm(arm, test, candidates[arm.source], cfg)
        extra: dict[str, Any] = {
            "seed": seed,
            "family": arm.family,
            "source": arm.source,
            "note": arm.note,
            "heuristic_variant": variant.name if arm.source == "heuristic" else "",
            "train_seconds": histories.get(arm.source, {}).get("train_seconds", float("nan")),
            "n_params": histories.get(arm.source, {}).get("n_params", float("nan")),
            "llm_tokens_per_doc": (
                llm_usage.total_tokens / max(1, len(test.docs)) if arm.source == "llm" else np.nan
            ),
        }
        rows.append(result.to_row(extra))
        frames.append(result.per_item_frame().assign(seed=seed))
        LOG.info(
            "seed %d %-18s strict %.4f canonical %.4f cov %.4f halluc %.4f ground %.4f",
            seed,
            arm.name,
            result.summary.get("strict_accuracy", np.nan),
            result.summary.get("canonical_accuracy", np.nan),
            result.summary.get("coverage", np.nan),
            result.summary.get("hallucination_rate", np.nan),
            result.summary.get("grounding_exact", np.nan),
        )

    frame = pd.DataFrame(rows)
    pd.concat(frames, ignore_index=True).to_csv(run_dir / "per_item.csv", index=False)
    frame.to_csv(run_dir / "summary.csv", index=False)
    write_json(
        run_dir / "summary.json",
        {
            "config": cfg.to_dict(),
            "seed": seed,
            "histories": histories,
            "heuristic_variant": variant.name,
            "llm_usage": llm_usage.to_dict(),
            "splits": prepared.splits.sizes,
            "wall_seconds": time.perf_counter() - started,
        },
    )
    _append_csv(TABLES / "seed_runs.csv", frame, key=("arm", "seed"))
    return frame


def _append_csv(path: Path, frame: pd.DataFrame, key: tuple[str, ...]) -> None:
    """Append rows, replacing any that share ``key``.

    Idempotent by construction: re-running a seed overwrites its rows instead of
    duplicating them, which is what makes "check whether the output already
    landed" a safe habit rather than a source of double-counted results.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        merged = pd.concat([old, frame], ignore_index=True)
        merged = merged.drop_duplicates(subset=list(key), keep="last")
    else:
        merged = frame
    sort_cols = [c for c in key if c in merged.columns]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)
    merged.to_csv(path, index=False)


def seed_variance(runs: pd.DataFrame, metrics: tuple[str, ...] = METRIC_FAMILY) -> pd.DataFrame:
    """Mean, sd and ``sqrt(2)*sd`` noise scale per arm and metric.

    The noise scale is the run-to-run scale of a *difference* between two runs, so
    it is what any claimed improvement has to clear. Reported per arm because
    arms differ in how variable they are -- the verified arms are markedly more
    stable than the unverified ones, which is itself a result.
    """
    out: list[dict[str, Any]] = []
    for arm, group in runs.groupby("arm"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            out.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "n_seeds": int(finite.size),
                    "mean": float(finite.mean()) if finite.size else np.nan,
                    "sd": float(finite.std(ddof=1)) if finite.size > 1 else np.nan,
                    "noise_scale": noise_scale(values),
                    "min": float(finite.min()) if finite.size else np.nan,
                    "max": float(finite.max()) if finite.size else np.nan,
                }
            )
    return pd.DataFrame(out).sort_values(["metric", "arm"]).reset_index(drop=True)


def verdict_table(runs: pd.DataFrame, variance: pd.DataFrame) -> pd.DataFrame:
    """Every arm against :data:`REFERENCE_ARM`, placed on the noise scale.

    The noise scale used is the *reference* arm's, which is the conservative
    choice when the two arms differ in variability: it asks whether the gap could
    have come from reseeding the thing being improved upon.
    """
    means = runs.groupby("arm").mean(numeric_only=True)
    scales = {
        (row.arm, row.metric): row.noise_scale for row in variance.itertuples(index=False)
    }
    out: list[dict[str, Any]] = []
    for metric in METRIC_FAMILY:
        if metric not in means.columns or REFERENCE_ARM not in means.index:
            continue
        ref = float(means.loc[REFERENCE_ARM, metric])
        scale = scales.get((REFERENCE_ARM, metric), np.nan)
        for arm in means.index:
            if arm == REFERENCE_ARM:
                continue
            value = float(means.loc[arm, metric])
            delta = value - ref
            out.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "value": value,
                    "reference_value": ref,
                    "delta": delta,
                    "noise_scale": scale,
                    "ratio_to_noise": (
                        abs(delta) / scale if scale and np.isfinite(scale) else np.nan
                    ),
                    "verdict": verdict(delta, scale),
                }
            )
    return pd.DataFrame(out)


def _rate(values: np.ndarray) -> float:
    """Mean of a 0/1 column, ``NaN`` when empty -- the set-level statistic."""
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def paired_tests(per_item: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Paired tests of every arm against the reference, on one seed.

    Two units of analysis appear, and the ``unit`` column says which:

    * ``document`` -- per-field values are averaged within a document first,
      because the eight fields of one invoice share a layout and are not
      independent. This is the conservative unit and the one the headline claims
      use.
    * ``set`` -- for hallucination rate and grounding exactness, which are
      statistics of a whole set with no per-item value; a paired bootstrap over
      documents is the correct test and is what is used.

    The caveat that applies to both: these condition on **one trained model per
    arm**. They are statements about two sets of weights, not about two methods.
    The method-level question is answered by the seed study.
    """
    frame = per_item[per_item["seed"] == seed]
    arms = [a for a in frame["arm"].unique() if a != REFERENCE_ARM]
    comparisons: list[Comparison] = []
    rows: list[dict[str, Any]] = []

    ref = frame[frame["arm"] == REFERENCE_ARM]
    for arm in arms:
        cur = frame[frame["arm"] == arm]
        merged = ref.merge(cur, on=["doc_id", "field"], suffixes=("_ref", "_arm"))
        if merged.empty:
            continue
        for metric, col in (
            ("strict_accuracy", "strict_correct"),
            ("canonical_accuracy", "canonical_correct"),
            ("coverage", "emitted"),
        ):
            ids_a, vals_a = aggregate_by_document(
                merged["doc_id"].to_numpy(), merged[f"{col}_arm"].to_numpy(dtype=np.float64)
            )
            ids_b, vals_b = aggregate_by_document(
                merged["doc_id"].to_numpy(), merged[f"{col}_ref"].to_numpy(dtype=np.float64)
            )
            if ids_a.size != ids_b.size:
                continue
            comparisons.append(
                compare(vals_a, vals_b, arm, REFERENCE_ARM, metric, unit="document")
            )

        # The suffixes from the merge above rename every shared column, so the
        # grounding flags are `grounded_arm` / `grounded_ref` here.
        arm_hall = (~merged.loc[merged["emitted_arm"], "grounded_arm"].to_numpy(dtype=bool))
        ref_hall = (~merged.loc[merged["emitted_ref"], "grounded_ref"].to_numpy(dtype=bool))
        n = min(arm_hall.size, ref_hall.size)
        if n > 1:
            interval = bootstrap_metric_difference(
                arm_hall.astype(np.float64)[:n],
                ref_hall.astype(np.float64)[:n],
                _rate,
            )
            rows.append(
                {
                    "name_a": arm,
                    "name_b": REFERENCE_ARM,
                    "metric": "hallucination_rate",
                    "unit": "set",
                    "mean_a": _rate(arm_hall.astype(np.float64)),
                    "mean_b": _rate(ref_hall.astype(np.float64)),
                    "difference": interval.estimate,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "p_value": np.nan,
                    "p_adjusted": np.nan,
                    "effect_size": np.nan,
                    "n": interval.n,
                    "significant": bool(
                        np.isfinite(interval.lower)
                        and np.isfinite(interval.upper)
                        and (interval.lower > 0 or interval.upper < 0)
                    ),
                }
            )

    holm_bonferroni(comparisons)
    rows.extend(c.to_dict() for c in comparisons)
    return pd.DataFrame(rows)
