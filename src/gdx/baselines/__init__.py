"""Real, runnable baselines: rule-based geometry, and an LLM pipeline arm."""

from __future__ import annotations

from gdx.baselines.heuristic import (
    VARIANT_BY_NAME,
    VARIANTS,
    HeuristicVariant,
    extract_candidates,
    extract_field,
    find_label_positions,
)
from gdx.baselines.llm import extract_candidates as llm_extract_candidates

__all__ = [
    "VARIANTS",
    "VARIANT_BY_NAME",
    "HeuristicVariant",
    "extract_candidates",
    "extract_field",
    "find_label_positions",
    "llm_extract_candidates",
]
