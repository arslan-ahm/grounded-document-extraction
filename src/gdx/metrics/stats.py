"""Statistical machinery for comparing extraction methods.

Reporting that one method reached 0.91 strict accuracy and another 0.89 is not a
result. Two different questions have to be kept apart, and conflating them is the
most common failure in small-scale ML reporting:

**Question 1 -- is the difference consistent across test items?** Answered by a
paired test over per-item values (:func:`compare`). It conditions on *one trained
model per method*, so it is a statement about two sets of weights, not about two
methods. Every table in this repository that uses it says so.

**Question 2 -- is the difference bigger than the run-to-run noise?** Answered by
training the same configuration under several seeds and comparing the difference
against ``sqrt(2) * sd`` (:func:`noise_scale`, :func:`verdict`). The sampling unit
is the *training run*. No number of test documents substitutes for it.

A wrinkle specific to this project: the natural unit of analysis is the
**(document, field) pair**, and the eight fields of one document are not
independent -- they share a layout, a currency style and a page count. A paired
test over pairs therefore overstates its own precision. Where a claim needs a
clean unit, :func:`aggregate_by_document` reduces to one value per document
first, and the tables say which unit was used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats


@dataclass
class Interval:
    """A point estimate with a confidence interval and its item count."""

    estimate: float
    lower: float
    upper: float
    level: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class Comparison:
    """The outcome of comparing two methods on the same items."""

    name_a: str
    name_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: Interval
    p_value: float
    effect_size: float
    n: int
    unit: str = "field"
    p_adjusted: float | None = None

    @property
    def significant(self) -> bool:
        p = self.p_value if self.p_adjusted is None else self.p_adjusted
        return bool(np.isfinite(p) and p < 0.05)

    def to_dict(self) -> dict[str, object]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "metric": self.metric,
            "unit": self.unit,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference.estimate,
            "ci_lower": self.difference.lower,
            "ci_upper": self.difference.upper,
            "p_value": self.p_value,
            "p_adjusted": self.p_adjusted,
            "effect_size": self.effect_size,
            "n": self.n,
            "significant": self.significant,
        }


def bootstrap_ci(
    values,  # noqa: ANN001
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
    statistic: str = "mean",
) -> Interval:
    """Percentile bootstrap interval for a summary of ``values``.

    ``NaN`` entries are dropped and the surviving count recorded, because an
    undefined per-item value must not be silently replaced by a convenient one.
    """
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    reduce = np.mean if statistic == "mean" else np.median
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0)
    point = float(reduce(data))
    if data.size < 2:
        return Interval(point, point, point, level, int(data.size))
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, data.size, size=(n_resamples, data.size))
    reps = reduce(data[picks], axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return Interval(point, float(lo), float(hi), level, int(data.size))


def paired_bootstrap_difference(
    a, b, n_resamples: int = 2000, level: float = 0.95, seed: int = 0  # noqa: ANN001
) -> Interval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples *item indices*, preserving the pairing. Items where either side is
    ``NaN`` are dropped before resampling, not after, so every replicate has the
    same support.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must match: {x.shape} vs {y.shape}")
    valid = np.isfinite(x) & np.isfinite(y)
    return bootstrap_ci((x - y)[valid], n_resamples, level, seed)


def compare(
    a,  # noqa: ANN001
    b,  # noqa: ANN001
    name_a: str = "a",
    name_b: str = "b",
    metric: str = "metric",
    unit: str = "field",
    n_resamples: int = 2000,
    seed: int = 0,
) -> Comparison:
    """Paired Wilcoxon signed-rank plus a paired bootstrap CI and effect size.

    ``p_value`` is ``NaN`` when every paired difference is exactly zero, because
    the signed-rank test is undefined there. That happens routinely in this
    project -- two arms that share an encoder and differ only in an inference-time
    switch agree exactly on most fields -- and reporting ``p = 1.0`` instead would
    be a claim the data does not make.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    diff = x - y
    if diff.size == 0 or np.allclose(diff, 0.0):
        p = float("nan")
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        unit=unit,
        mean_a=float(x.mean()) if x.size else float("nan"),
        mean_b=float(y.mean()) if y.size else float("nan"),
        difference=paired_bootstrap_difference(x, y, n_resamples, seed=seed),
        p_value=p,
        effect_size=float(diff.mean() / sd) if sd > 0 else 0.0,
        n=int(diff.size),
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Holm-Bonferroni step-down correction, applied in place.

    ``NaN`` p-values are excluded from the family size, since an undefined test
    carries no evidence in either direction and inflating ``m`` with it would
    make every other test in the family harder to pass for no reason.
    """
    del alpha
    testable = [c for c in comparisons if np.isfinite(c.p_value)]
    m = len(testable)
    if m == 0:
        return comparisons
    order = sorted(range(m), key=lambda i: testable[i].p_value)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * testable[idx].p_value)
        running = max(running, adjusted)
        testable[idx].p_adjusted = running
    return comparisons


def bootstrap_metric_difference(
    items_a,  # noqa: ANN001
    items_b,  # noqa: ANN001
    metric: Callable[[np.ndarray], float],
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Paired bootstrap for the difference in a *set-level* statistic.

    Hallucination rate, AURC and error-detection AUROC are functions of a whole
    set, not per-item values, so there is nothing for a Wilcoxon test to consume.
    The correct paired test resamples item indices and recomputes the statistic on
    each resample for both methods, which is what this does.

    Args:
        items_a: ``(N, ...)`` per-item data for method A.
        items_b: ``(N, ...)`` per-item data for method B, same order.
        metric: Maps resampled per-item data to a scalar.
        n_resamples: Replicates.
        level: Coverage.
        seed: RNG seed.

    Returns:
        Interval on ``metric(A) - metric(B)``. Replicates where the statistic is
        undefined for either method are dropped and the surviving count reported.
    """
    a = np.asarray(items_a, dtype=np.float64)
    b = np.asarray(items_b, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"paired data must match on axis 0: {a.shape} vs {b.shape}")
    point = metric(a) - metric(b)
    rng = np.random.default_rng(seed)
    reps: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, a.shape[0], size=a.shape[0])
        value = metric(a[idx]) - metric(b[idx])
        if np.isfinite(value):
            reps.append(float(value))
    if not reps:
        return Interval(float(point), float("nan"), float("nan"), level, 0)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(reps, [alpha, 1.0 - alpha])
    return Interval(float(point), float(lo), float(hi), level, len(reps))


def noise_scale(values) -> float:  # noqa: ANN001
    """Run-to-run scale of a *difference*, from repeated-seed values.

    Two independent runs differ with standard deviation ``sqrt(2) * sd``, so a
    gap smaller than that is not distinguishable from having reseeded.

    Args:
        values: The metric from >= 2 runs of the *same* configuration.

    Returns:
        ``sqrt(2) * sd``, or ``NaN`` with fewer than two finite values.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    return float(np.sqrt(2.0) * v.std(ddof=1))


def verdict(difference: float, scale: float) -> str:
    """Classify a claimed improvement against the run-to-run noise scale.

    Returns ``"robust"`` (>= 3x), ``"survives"`` (>= 2x), ``"suggestive"``
    (>= 1x), ``"inside noise"`` (< 1x), or ``"unknown"`` when the scale could not
    be estimated. The thresholds are a reporting convention, stated here so a
    reader can apply their own.
    """
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(difference):
        return "unknown"
    ratio = abs(difference) / scale
    if ratio >= 3.0:
        return "robust"
    if ratio >= 2.0:
        return "survives"
    if ratio >= 1.0:
        return "suggestive"
    return "inside noise"


def aggregate_by_document(doc_ids, values) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Reduce per-field values to one value per document.

    The eight fields of a document share its layout and are therefore correlated;
    a paired test over fields treats them as independent and reports a narrower
    interval than the data supports. Reducing to the document mean first gives a
    conservative, defensible unit of analysis.

    Returns:
        ``(unique_doc_ids, mean_value_per_document)``, sorted by id. Documents
        with no finite value are dropped.
    """
    ids = np.asarray(doc_ids)
    vals = np.asarray(values, dtype=np.float64)
    if ids.shape != vals.shape:
        raise ValueError(f"shape mismatch: {ids.shape} vs {vals.shape}")
    uniq = np.unique(ids)
    out_ids: list = []
    out_vals: list[float] = []
    for key in uniq:
        picked = vals[ids == key]
        picked = picked[np.isfinite(picked)]
        if picked.size:
            out_ids.append(key)
            out_vals.append(float(picked.mean()))
    return np.asarray(out_ids), np.asarray(out_vals, dtype=np.float64)
