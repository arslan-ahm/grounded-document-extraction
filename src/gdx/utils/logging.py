"""Console logging and append-only JSONL run history.

Every training run writes ``history.jsonl`` next to its config and summary. One
JSON object per line, appended and flushed immediately, so a run that is killed
half-way still leaves a readable partial history -- which matters on a machine
where jobs get interrupted.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False


def get_logger(name: str = "gdx", level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing to stdout with a compact format.

    Configures the root handler exactly once; repeated calls from different
    modules do not duplicate output.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("gdx")
        root.handlers = [handler]
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger


def _jsonable(value: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts.

    Non-finite floats become ``None`` rather than the ``NaN`` literal, which is
    not valid JSON and breaks strict parsers downstream.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


class JsonlLogger:
    """Append-only JSONL writer for per-epoch training records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []

    def log(self, **fields: Any) -> dict[str, Any]:
        """Append one record and return it."""
        record = {k: _jsonable(v) for k, v in fields.items()}
        self._records.append(record)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @staticmethod
    def read(path: str | Path) -> list[dict[str, Any]]:
        """Read a JSONL history back, skipping blank and malformed lines."""
        out: list[dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as pretty JSON with LF endings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    p.write_text(text, encoding="utf-8", newline="\n")
    return p
