"""Put ``src`` on the path and cap CPU threads. Imported by every script.

Thread capping happens at import, before torch builds its thread pool, because
setting it afterwards is ignored by some BLAS backends -- and an uncapped run
both slows the shared machine down and makes every latency number in
``results/tables/`` wrong.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

__all__ = ["ROOT"]
