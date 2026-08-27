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

from gdx.pipelines.analysis import write_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    written = write_all(args.seeds)
    for name, path in written.items():
        frame = pd.read_csv(path)
        print(f"\n=== {name} ({len(frame)} rows) ===")
        print(frame.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
