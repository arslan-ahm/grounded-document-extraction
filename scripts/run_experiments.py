"""Run the method-comparison matrix for one seed, or several.

    python scripts/run_experiments.py --seed 0
    python scripts/run_experiments.py --seeds 0 1 2      # sequential
    python scripts/run_experiments.py --seed 1 --force   # redo a landed seed

Each seed trains both heads, evaluates all seven arms and appends its rows to
``results/tables/seed_runs.csv``. One invocation per seed is the intended usage:
a crash then costs one seed, and an already-completed seed is skipped unless
``--force`` is given.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from gdx.config import load_config
from gdx.pipelines.experiments import TABLES, run_seed


def already_done(seed: int) -> bool:
    path = TABLES / "seed_runs.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return bool((frame["seed"] == seed).sum() >= 7)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None else 0]
    cfg = load_config(args.config, args.overrides)
    for seed in seeds:
        if already_done(seed) and not args.force:
            print(f"seed {seed} already in results/tables/seed_runs.csv; skipping")
            continue
        frame = run_seed(cfg, seed)
        print(f"\nseed {seed}:")
        cols = ["arm", "strict_accuracy", "canonical_accuracy", "coverage",
                "hallucination_rate", "grounding_exact"]
        print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
