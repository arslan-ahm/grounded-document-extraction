"""Draw every figure whose source CSV exists.

    python scripts/make_figures.py

Figures are functions of the committed tables, so a figure cannot show something
the tables do not. Missing sources are skipped with a message rather than
producing an empty plot.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gdx.viz import make_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    written = make_all(seed=args.seed)
    if not written:
        print("no figures written; run the experiments first")
        return 1
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
