"""Console entry point: ``gdx <command>``.

A thin dispatcher over the same pipeline functions the scripts use, so the two
cannot drift. Every command takes ``--config`` and ``--set`` and behaves exactly
like its script counterpart.
"""

from __future__ import annotations

import argparse
import sys

from gdx import __version__
from gdx.config import load_config
from gdx.utils.seed import limit_threads


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gdx", description=__doc__)
    parser.add_argument("--version", action="version", version=f"gdx {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train one head and evaluate its arms")
    _common(train)

    experiments = sub.add_parser("experiments", help="run the matrix for one seed")
    _common(experiments)
    experiments.add_argument("--seed", type=int, default=0)

    ablations = sub.add_parser("ablations", help="run the ablation matrix for one seed")
    _common(ablations)
    ablations.add_argument("--seed", type=int, default=0)

    bench = sub.add_parser("benchmark", help="measure params, MACs, latency, token cost")
    _common(bench)

    analyse = sub.add_parser("analyse", help="recompute derived tables from the CSVs")
    analyse.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    figures = sub.add_parser("figures", help="draw every figure whose CSV exists")
    figures.add_argument("--seed", type=int, default=0)

    data = sub.add_parser("data", help="print a generated document as text")
    _common(data)
    data.add_argument("--doc-id", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    limit_threads(2)

    if args.command == "analyse":
        from gdx.pipelines.analysis import write_all

        for name, path in write_all(args.seeds).items():
            print(f"{name}: {path}")
        return 0

    if args.command == "figures":
        from gdx.viz import make_all

        for name, path in make_all(seed=args.seed).items():
            print(f"{name}: {path}")
        return 0

    cfg = load_config(args.config, args.overrides)

    if args.command == "train":
        from gdx.pipelines.core import run_single

        run_single(cfg)
        print(f"run directory: {cfg.run_dir}")
        return 0

    if args.command == "experiments":
        from gdx.pipelines.experiments import run_seed

        run_seed(cfg, args.seed)
        return 0

    if args.command == "ablations":
        from gdx.pipelines.ablations import (
            run_model_ablations,
            run_verify_ablations,
            summarise_ablations,
            train_shared_span,
        )

        shared = train_shared_span(cfg, args.seed)
        run_verify_ablations(cfg, args.seed, shared=shared)
        run_model_ablations(cfg, args.seed, shared=shared)
        summarise_ablations()
        return 0

    if args.command == "benchmark":
        from gdx.pipelines.core import prepare
        from gdx.pipelines.efficiency import baseline_cost, decode_scaling, head_cost
        from gdx.pipelines.experiments import TABLES

        prepared = prepare(cfg, seed=0)
        TABLES.mkdir(parents=True, exist_ok=True)
        head_cost(cfg, prepared).to_csv(TABLES / "efficiency.csv", index=False)
        decode_scaling(cfg, prepared).to_csv(TABLES / "decode_scaling.csv", index=False)
        baseline_cost(cfg, prepared).to_csv(TABLES / "baseline_cost.csv", index=False)
        print(f"wrote efficiency tables to {TABLES}")
        return 0

    if args.command == "data":
        from gdx.data.generator import generate_document

        doc = generate_document(args.doc_id, cfg.data, seed=cfg.run.seed)
        print(f"document {doc.doc_id}: {len(doc)} tokens, {doc.n_pages} page(s)")
        print(" ".join(doc.texts))
        print()
        for name, truth in doc.fields.items():
            state = "absent" if not truth.present else f"{truth.span} {truth.span_text!r}"
            flag = " (needs normalisation)" if truth.requires_normalisation else ""
            print(f"  {name:<14s} {state}{flag}")
        return 0

    print(f"unknown command {args.command!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
