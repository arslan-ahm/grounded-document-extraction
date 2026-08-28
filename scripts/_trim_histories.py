"""One-off repair: keep only the last run's records in a history.jsonl.

Needed because an earlier version of the trainer appended to an existing history
when a seed was re-run, so a re-run seed's file holds two runs spliced together.
The trainer now truncates; this fixes files written before that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def trim(path: Path) -> int:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    starts = [i for i, r in enumerate(records) if r.get("epoch") == 0]
    if len(starts) <= 1:
        return 0
    kept = records[starts[-1]:]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(records) - len(kept)


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("results/runs")
    for path in sorted(root.glob("**/history.jsonl")):
        dropped = trim(path)
        print(f"{path}: dropped {dropped} stale record(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
