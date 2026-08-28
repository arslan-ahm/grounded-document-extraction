"""Recompute every derived table from the committed per-item CSVs.

    python scripts/analyse.py --seeds 0 1 2

Runs no model. Everything it produces is a function of
``results/runs/seed*/per_item.csv`` and ``results/tables/seed_runs.csv``, so a
reader can regenerate the statistics without retraining anything.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from gdx.config import load_config
from gdx.pipelines.analysis import heuristic_sweep_table, write_all
from gdx.pipelines.experiments import TABLES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_sweep:
        # The rule baseline's configuration sweep is re-run here rather than
        # cached from the training run, so the committed table is reproducible
        # from the config alone.
        sweep = heuristic_sweep_table(load_config(args.config), seed=args.seeds[0])
        sweep.to_csv(TABLES / "heuristic_sweep.csv", index=False)
    written = write_all(args.seeds)
    for name, path in written.items():
        frame = pd.read_csv(path)
        print(f"\n=== {name} ({len(frame)} rows) ===")
        print(frame.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
