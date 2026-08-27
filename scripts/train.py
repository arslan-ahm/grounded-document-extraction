"""Train one head and evaluate the arms that share it.

    python scripts/train.py --config configs/smoke.yaml
    python scripts/train.py --config configs/span.yaml --set optim.epochs=10

Writes ``results/runs/<run.name>/`` containing the resolved ``config.yaml``, the
per-epoch ``history.jsonl``, ``per_item.csv`` and ``summary.json``. Thin by
design: all logic lives in :mod:`gdx.pipelines.core` so the tests can reach it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gdx.config import load_config  # noqa: E402
from gdx.pipelines.core import run_single  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/span.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    payload = run_single(cfg)
    print(f"\nrun directory: {cfg.run_dir}")
    for row in payload["arms"]:
        print(
            f"  {row['arm']:<18s} strict={row['strict_accuracy']:.4f} "
            f"canonical={row['canonical_accuracy']:.4f} "
            f"coverage={row['coverage']:.4f} "
            f"hallucination={row['hallucination_rate']:.4f} "
            f"n={int(row['n_records'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
