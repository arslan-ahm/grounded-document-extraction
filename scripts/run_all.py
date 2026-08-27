"""Run the whole experiment matrix, in the order the results are read in.

    python scripts/run_all.py                 # everything, ~90 minutes on 2 threads
    python scripts/run_all.py --stage bench   # one stage
    python scripts/run_all.py --seeds 0 1 2

Stages, in order:

1. ``bench``       -- params, MACs, latency, decode scaling, token cost
2. ``experiments`` -- the seven-arm comparison, one invocation per seed
3. ``ablations``   -- inference-time and training-time switches, per seed
4. ``analyse``     -- every derived table, from the committed per-item CSVs
5. ``determinism`` -- run one evaluation twice and record the max difference
6. ``figures``     -- every figure whose source CSV exists
7. ``docs``        -- inject the tables into the Markdown

Each stage checks whether its output already landed and skips it unless
``--force`` is given, so an interrupted run is resumed rather than restarted.
"""

from __future__ import annotations

import argparse

import _bootstrap
import pandas as pd

from gdx.config import load_config
from gdx.pipelines.experiments import TABLES

STAGES = ("bench", "experiments", "ablations", "analyse", "determinism", "figures", "docs")


def _have(name: str, min_rows: int = 1) -> bool:
    path = TABLES / f"{name}.csv"
    return path.exists() and len(pd.read_csv(path)) >= min_rows


def _seed_landed(table: str, seed: int, min_rows: int) -> bool:
    if not _have(table):
        return False
    frame = pd.read_csv(TABLES / f"{table}.csv")
    return bool((frame["seed"] == seed).sum() >= min_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--stage", choices=list(STAGES), default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    stages = [args.stage] if args.stage else list(STAGES)

    if "bench" in stages and (args.force or not _have("efficiency")):
        from gdx.pipelines.core import prepare
        from gdx.pipelines.efficiency import baseline_cost, decode_scaling, head_cost

        prepared = prepare(cfg, seed=0)
        TABLES.mkdir(parents=True, exist_ok=True)
        head_cost(cfg, prepared).to_csv(TABLES / "efficiency.csv", index=False)
        decode_scaling(cfg, prepared).to_csv(TABLES / "decode_scaling.csv", index=False)
        baseline_cost(cfg, prepared).to_csv(TABLES / "baseline_cost.csv", index=False)

    if "experiments" in stages:
        from gdx.pipelines.experiments import run_seed

        for seed in args.seeds:
            if _seed_landed("seed_runs", seed, 7) and not args.force:
                print(f"seed {seed} already landed; skipping")
                continue
            run_seed(cfg, seed)

    if "ablations" in stages:
        from gdx.pipelines.ablations import (
            run_model_ablations,
            run_verify_ablations,
            train_shared_span,
        )

        for seed in args.seeds:
            if _seed_landed("ablation_runs", seed, 8) and not args.force:
                print(f"ablation seed {seed} already landed; skipping")
                continue
            shared = train_shared_span(cfg, seed)
            run_verify_ablations(cfg, seed, shared=shared)
            run_model_ablations(cfg, seed, shared=shared)

    if "analyse" in stages:
        from gdx.pipelines.ablations import summarise_ablations
        from gdx.pipelines.analysis import write_all

        write_all(args.seeds)
        if _have("ablation_runs"):
            summarise_ablations()

    if "determinism" in stages and (args.force or not _have("determinism")):
        from gdx.pipelines.determinism import check_determinism

        check_determinism(cfg, seed=args.seeds[0])

    if "figures" in stages:
        from gdx.viz import make_all

        make_all(seed=args.seeds[0])

    if "docs" in stages:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(_bootstrap.ROOT / "scripts" / "render_docs.py")], check=False
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
