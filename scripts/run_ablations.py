"""Run the ablation matrix for one seed.

    python scripts/run_ablations.py --seed 0
    python scripts/run_ablations.py --seed 0 --kind inference

Inference-time ablations reuse one span checkpoint; training-time ablations
retrain. Rows are appended to ``results/tables/ablation_runs.csv``, and
``--summarise`` recomputes ``results/tables/ablations.csv`` from whatever has
landed.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gdx.config import load_config
from gdx.pipelines.ablations import (
    run_model_ablations,
    run_verify_ablations,
    summarise_ablations,
    train_shared_span,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kind", choices=["inference", "training", "both"], default="both")
    parser.add_argument("--summarise", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    # One shared training run backs the inference ablations and the training
    # ablations' `full` row, so the seed is not paid for twice.
    shared = None if args.summarise and args.kind == "none" else train_shared_span(cfg, args.seed)
    if args.kind in ("inference", "both"):
        run_verify_ablations(cfg, args.seed, shared=shared)
    if args.kind in ("training", "both"):
        run_model_ablations(cfg, args.seed, shared=shared)
    if args.summarise:
        print(summarise_ablations().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
