"""Ablations: one config switch each, with the noise scale attached.

Two kinds, and keeping them apart matters.

**Inference-time switches** -- verification on/off, type check on/off, arithmetic
check on/off, abstention on/off. These reuse the *same trained weights*, so the
comparison contains no initialisation noise at all and the delta is attributable
to the switch alone. Retraining for each would confound the mechanism with the
seed.

**Training-time switches** -- 2-D layout position on/off, spatial attention bias
on/off. These require a retrain, so their deltas carry run-to-run noise and are
placed against ``sqrt(2)*sd`` from the seed study before being interpreted.

Every ablation flips exactly one field of the config. A variant that changed two
things would attribute nothing.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gdx.arms import ARM_BY_NAME, Arm, model_candidates
from gdx.config import Config
from gdx.metrics.stats import noise_scale, verdict
from gdx.pipelines.core import evaluate_arm, prepare, train_head
from gdx.pipelines.experiments import TABLES, _append_csv
from gdx.utils.logging import get_logger

LOG = get_logger(__name__)

#: Inference-time ablations. Each is a single override on ``cfg.verify``,
#: evaluated against the same span checkpoint.
VERIFY_ABLATIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("full", {}),
    ("no_verification", {"enabled": False}),
    ("no_type_check", {"type_check": False}),
    ("no_arithmetic", {"arithmetic": False}),
    ("no_abstain", {"abstain": False}),
    ("conf_threshold_0.5", {"min_confidence": 0.5}),
)

#: Training-time ablations. Each is a single override on ``cfg.model``.
MODEL_ABLATIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("full", {}),
    ("no_2d_pos", {"use_2d_pos": False}),
    ("no_spatial_bias", {"use_spatial_bias": False}),
)


def run_verify_ablations(
    cfg: Config, seed: int, shared: tuple[Any, dict[str, Any], Any, list] | None = None
) -> pd.DataFrame:
    """All inference-time ablations on one span checkpoint.

    Trains once, decodes once, then re-runs only the verification loop under each
    switch. The candidate sets are byte-identical across rows by construction.

    Args:
        cfg: Configuration.
        seed: Seed.
        shared: An already-trained ``(model, history, splits_test, candidates)``
            tuple, so the training-time ablation's ``full`` arm and this one can
            share a single ~3-minute training run instead of paying for it twice.
    """
    if shared is None:
        prepared = prepare(cfg, seed=seed)
        model, history = train_head(cfg, prepared, "span", None)
        test = prepared.splits.test
        candidates = model_candidates(model, test, cfg)
    else:
        model, history, test, candidates = shared
    del model

    rows: list[dict[str, Any]] = []
    for name, overrides in VERIFY_ABLATIONS:
        arm = Arm(f"verify:{name}", "span", "ablation", overrides, note=str(overrides))
        result = evaluate_arm(arm, test, candidates, cfg)
        rows.append(
            result.to_row(
                {
                    "seed": seed,
                    "ablation": name,
                    "kind": "inference",
                    "switch": ", ".join(f"{k}={v}" for k, v in overrides.items()) or "none",
                    "train_seconds": history.get("train_seconds", np.nan),
                }
            )
        )
        LOG.info(
            "seed %d verify:%-18s strict %.4f cov %.4f halluc %.4f arith_ok %.4f",
            seed, name,
            result.summary.get("strict_accuracy", np.nan),
            result.summary.get("coverage", np.nan),
            result.summary.get("hallucination_rate", np.nan),
            result.summary.get("verify_arith_ok_frac", np.nan),
        )
    frame = pd.DataFrame(rows)
    _append_csv(TABLES / "ablation_runs.csv", frame, key=("arm", "seed"))
    return frame


def run_model_ablations(
    cfg: Config, seed: int, shared: tuple[Any, dict[str, Any], Any, list] | None = None
) -> pd.DataFrame:
    """Training-time ablations: one retrain each, evaluated with the full loop.

    ``shared`` supplies an already-trained full model so the ``full`` row does not
    pay for a second identical training run.
    """
    rows: list[dict[str, Any]] = []
    arm = ARM_BY_NAME["span_verify"]
    for name, overrides in MODEL_ABLATIONS:
        started = time.perf_counter()
        local = replace(cfg, model=replace(cfg.model, head="span", **overrides))
        if name == "full" and shared is not None:
            model, history, test, candidates = shared
        else:
            prepared = prepare(local, seed=seed)
            model, history = train_head(local, prepared, "span", None)
            test = prepared.splits.test
            candidates = model_candidates(model, test, local)
        result = evaluate_arm(arm, test, candidates, local)
        rows.append(
            result.to_row(
                {
                    "arm": f"model:{name}",
                    "seed": seed,
                    "ablation": name,
                    "kind": "training",
                    "switch": ", ".join(f"{k}={v}" for k, v in overrides.items()) or "none",
                    "train_seconds": history.get("train_seconds", np.nan),
                    "n_params": history.get("n_params", np.nan),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        )
        LOG.info(
            "seed %d model:%-18s strict %.4f canonical %.4f ground %.4f params %s",
            seed, name,
            result.summary.get("strict_accuracy", np.nan),
            result.summary.get("canonical_accuracy", np.nan),
            result.summary.get("grounding_exact", np.nan),
            history.get("n_params"),
        )
    frame = pd.DataFrame(rows)
    frame["arm"] = [f"model:{n}" for n, _ in MODEL_ABLATIONS]
    _append_csv(TABLES / "ablation_runs.csv", frame, key=("arm", "seed"))
    return frame


def summarise_ablations(
    metrics: tuple[str, ...] = (
        "strict_accuracy",
        "canonical_accuracy",
        "coverage",
        "grounding_exact",
        "hallucination_rate",
    ),
    tables_dir: Path = TABLES,
) -> pd.DataFrame:
    """Ablation deltas against ``full``, each divided by its own noise scale.

    The noise scale for an *inference* ablation is computed from the ``full``
    row's seed-to-seed spread, which is the right reference even though the
    ablation itself adds no noise: the question is whether the switch's effect is
    larger than the variability of the pipeline it sits inside.
    """
    frame = pd.read_csv(tables_dir / "ablation_runs.csv")
    rows: list[dict[str, Any]] = []
    for kind, group in frame.groupby("kind"):
        base = group[group["ablation"] == "full"]
        if base.empty:
            continue
        for metric in metrics:
            if metric not in group.columns:
                continue
            scale = noise_scale(base[metric].to_numpy(dtype=np.float64))
            ref = float(np.nanmean(base[metric].to_numpy(dtype=np.float64)))
            for ablation, sub in group.groupby("ablation"):
                if ablation == "full":
                    continue
                value = float(np.nanmean(sub[metric].to_numpy(dtype=np.float64)))
                delta = value - ref
                rows.append(
                    {
                        "kind": kind,
                        "ablation": ablation,
                        "switch": str(sub["switch"].iloc[0]),
                        "metric": metric,
                        "value": value,
                        "full_value": ref,
                        "delta": delta,
                        "noise_scale": scale,
                        "ratio_to_noise": (
                            abs(delta) / scale if scale and np.isfinite(scale) else np.nan
                        ),
                        "verdict": verdict(delta, scale),
                        "n_seeds": int(len(sub)),
                    }
                )
    out = pd.DataFrame(rows).sort_values(["kind", "metric", "ablation"]).reset_index(drop=True)
    out.to_csv(tables_dir / "ablations.csv", index=False)
    return out


def train_shared_span(cfg: Config, seed: int):  # noqa: ANN201
    """Train the full span model once and return everything the ablations need.

    Returns ``(model, history, test_split, candidates)``. Sharing this across both
    ablation families removes a redundant training run per seed, which on this
    machine is three minutes of the project's two-hour compute budget.
    """
    prepared = prepare(cfg, seed=seed)
    model, history = train_head(cfg, prepared, "span", None)
    test = prepared.splits.test
    candidates = model_candidates(model, test, cfg)
    return model, history, test, candidates
