"""Measure params, MACs, latency and token cost; write the efficiency tables.

    python scripts/benchmark_efficiency.py

Writes ``results/tables/efficiency.csv``, ``decode_scaling.csv`` and
``baseline_cost.csv``. Uses untrained weights: latency does not depend on the
values in the tensors, and training first would cost minutes for no change to
the measurement.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from gdx.config import load_config
from gdx.pipelines.core import prepare
from gdx.pipelines.efficiency import (
    baseline_cost,
    decode_scaling,
    head_cost,
    torch_thread_note,
)
from gdx.pipelines.experiments import TABLES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    prepared = prepare(cfg, seed=0)
    TABLES.mkdir(parents=True, exist_ok=True)

    head = head_cost(cfg, prepared)
    head.to_csv(TABLES / "efficiency.csv", index=False)
    print(head.to_string(index=False))

    scaling = decode_scaling(cfg, prepared)
    scaling.to_csv(TABLES / "decode_scaling.csv", index=False)
    print("\n" + scaling.to_string(index=False))

    cost = baseline_cost(cfg, prepared)
    cost.to_csv(TABLES / "baseline_cost.csv", index=False)
    print("\n" + cost.to_string(index=False))

    (TABLES / "efficiency_env.json").write_text(
        json.dumps(torch_thread_note(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
