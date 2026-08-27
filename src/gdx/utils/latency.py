"""Wall-clock benchmarking.

Two details decide whether a latency table means anything, and both were
learned the hard way in a sibling project:

1. **Warm-up.** The first handful of calls pay for lazy kernel selection,
   allocator growth and page faults. With two warm-up iterations a tiny model
   measured ~5x slower than its steady state. The default here is 8, and the
   shipped benchmarks use more.
2. **Median and IQR, not mean.** On a shared 4-core machine another process's
   burst lands in the tail. The mean absorbs it; the median does not. The IQR is
   reported so a reader can see the spread rather than trusting a point.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass
class LatencyResult:
    """Timing summary in milliseconds."""

    median_ms: float
    iqr_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    n_repeats: int
    n_warmup: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def benchmark_callable(
    fn: Callable[[], object],
    n_warmup: int = 8,
    n_repeats: int = 25,
) -> LatencyResult:
    """Time ``fn`` with warm-up, returning median and IQR in milliseconds.

    Args:
        fn: Zero-argument callable. Anything it returns is discarded.
        n_warmup: Untimed calls first. Must be >= 1; the project standard asks
            for >= 8 in shipped numbers.
        n_repeats: Timed calls. Must be >= 2 for an IQR to exist.

    Returns:
        A :class:`LatencyResult`.

    Raises:
        ValueError: If ``n_repeats`` < 1 or ``n_warmup`` < 0.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be >= 0, got {n_warmup}")

    for _ in range(n_warmup):
        fn()

    samples: list[float] = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples_sorted = sorted(samples)
    if len(samples_sorted) >= 4:
        q1, _, q3 = statistics.quantiles(samples_sorted, n=4)
        iqr = float(q3 - q1)
    else:
        iqr = float("nan")
    return LatencyResult(
        median_ms=float(statistics.median(samples_sorted)),
        iqr_ms=iqr,
        min_ms=float(samples_sorted[0]),
        max_ms=float(samples_sorted[-1]),
        mean_ms=float(statistics.fmean(samples_sorted)),
        n_repeats=int(n_repeats),
        n_warmup=int(n_warmup),
    )


def fit_power_exponent(sizes: list[float], times: list[float]) -> float:
    """Least-squares exponent ``b`` in ``t = a * n**b``, fitted in log space.

    Used to substantiate asymptotic claims with measurements instead of
    assertions: the sequential decode cost of a generative head should come out
    near 1 in value length, and attention near 2 in sequence length.

    Returns:
        The fitted exponent, or ``NaN`` with fewer than two positive pairs.
    """
    import math

    pairs = [(s, t) for s, t in zip(sizes, times, strict=False) if s > 0 and t > 0]
    if len(pairs) < 2:
        return float("nan")
    xs = [math.log(s) for s, _ in pairs]
    ys = [math.log(t) for _, t in pairs]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 0:
        return float("nan")
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom)
