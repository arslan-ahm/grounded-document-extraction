"""Determinism evidence: run the same evaluation twice and diff every item.

The project standard asks for max absolute difference 0.0 across per-item scores
from two separate invocations. This module produces that number and writes it to
``results/tables/determinism.csv`` so ``docs/REPRODUCIBILITY.md`` quotes a
measurement rather than an assurance.

The check covers the two places nondeterminism could enter: the *generator* -- a
document must be a pure function of its id, seed and config -- and the
*evaluation*, where a trained model's per-item outputs must not move between two
decodes of the same weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gdx.arms import ARM_BY_NAME, model_candidates
from gdx.config import Config
from gdx.data.generator import generate_dataset
from gdx.pipelines.core import evaluate_arm, prepare, train_head
from gdx.pipelines.experiments import TABLES
from gdx.utils.logging import get_logger

LOG = get_logger(__name__)


def check_determinism(cfg: Config, seed: int = 0) -> pd.DataFrame:
    """Two invocations, one table of maximum absolute differences.

    Returns:
        A frame with one row per checked quantity: the item count, the maximum
        absolute difference, and whether the two runs were bit-identical. A
        string-valued quantity reports ``0.0`` when identical and ``NaN``
        otherwise, because there is no metric on strings and a fabricated one
        would be worse than an honest gap.
    """
    rows: list[dict[str, Any]] = []

    first = generate_dataset(30, cfg.data, seed=seed)
    second = generate_dataset(30, cfg.data, seed=seed)
    boxes_a = np.array([t.box for d in first for t in d.tokens], dtype=np.float64)
    boxes_b = np.array([t.box for d in second for t in d.tokens], dtype=np.float64)
    rows.append(
        {
            "quantity": "generator_token_boxes",
            "n": int(boxes_a.size),
            "max_abs_difference": float(np.abs(boxes_a - boxes_b).max()),
            "identical": bool(np.array_equal(boxes_a, boxes_b)),
        }
    )
    texts_a = [t.text for d in first for t in d.tokens]
    texts_b = [t.text for d in second for t in d.tokens]
    rows.append(
        {
            "quantity": "generator_token_texts",
            "n": len(texts_a),
            "max_abs_difference": 0.0 if texts_a == texts_b else float("nan"),
            "identical": texts_a == texts_b,
        }
    )

    prepared = prepare(cfg, seed=seed)
    model, _ = train_head(cfg, prepared, "span", None)
    arm = ARM_BY_NAME["span_verify"]
    scores: list[np.ndarray] = []
    values: list[list[str]] = []
    for _ in range(2):
        candidates = model_candidates(model, prepared.splits.test, cfg)
        result = evaluate_arm(arm, prepared.splits.test, candidates, cfg)
        scores.append(np.array([r.confidence for r in result.records], dtype=np.float64))
        values.append([r.predicted_value for r in result.records])
    diff = np.abs(np.nan_to_num(scores[0], nan=0.0) - np.nan_to_num(scores[1], nan=0.0))
    rows.append(
        {
            "quantity": "span_verify_confidence",
            "n": int(scores[0].size),
            "max_abs_difference": float(diff.max()) if diff.size else float("nan"),
            "identical": bool(np.array_equal(scores[0], scores[1], equal_nan=True)),
        }
    )
    rows.append(
        {
            "quantity": "span_verify_emitted_values",
            "n": len(values[0]),
            "max_abs_difference": 0.0 if values[0] == values[1] else float("nan"),
            "identical": values[0] == values[1],
        }
    )

    frame = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "determinism.csv", index=False)
    LOG.info("determinism:\n%s", frame.to_string(index=False))
    return frame
