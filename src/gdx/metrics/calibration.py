"""Calibration and selective prediction.

The question these metrics answer is not "how often is the model right" but
"does the model know when it is wrong", which is the question that matters for a
system that is allowed to abstain.

Two families:

**Calibration** -- ECE, equal-mass ACE, MCE, Brier, NLL. ECE with equal-width
bins is the number everyone quotes and it is the weakest of the set: with
confidences piled near 1.0, most bins are nearly empty and the statistic is
dominated by a handful of items. Equal-mass ACE (Nixon et al., 2019) puts the
same count in every bin and is reported alongside for exactly that reason. Both
are shown so a reader can see the disagreement rather than take one on trust.

**Selective prediction** -- the risk-coverage curve, its area (AURC), and
error-detection AUROC. AURC is the honest summary of an abstention policy: it
integrates error rate over every coverage level rather than reporting one
operating point that could have been chosen after the fact.

``NaN`` discipline throughout: a bin with no items has no accuracy, an AUROC with
only one class present is undefined, and both return ``NaN`` rather than a
convenient number.
"""

from __future__ import annotations

import numpy as np


def _clean(confidence, correct) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Drop pairs where the confidence is not finite, keeping them aligned."""
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape:
        raise ValueError(f"shape mismatch: {conf.shape} vs {corr.shape}")
    keep = np.isfinite(conf) & np.isfinite(corr)
    return conf[keep], corr[keep]


def expected_calibration_error(
    confidence, correct, n_bins: int = 10  # noqa: ANN001
) -> float:
    """ECE with ``n_bins`` equal-width bins on ``[0, 1]``.

    Returns ``NaN`` with no finite items. Empty bins contribute nothing, which is
    correct but is also why this statistic is unstable on peaked confidence
    distributions -- see :func:`adaptive_calibration_error`.
    """
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not in_bin.any():
            continue
        weight = in_bin.mean()
        total += weight * abs(corr[in_bin].mean() - conf[in_bin].mean())
    return float(total)


def adaptive_calibration_error(
    confidence, correct, n_bins: int = 10  # noqa: ANN001
) -> float:
    """Equal-mass ACE: quantile bins, so every bin has the same count.

    Preferred over ECE whenever confidences are concentrated, which is always the
    case for a trained extractor. With fewer items than bins the bin count is
    reduced rather than producing empty bins.
    """
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    bins = int(min(n_bins, conf.size))
    order = np.argsort(conf, kind="stable")
    chunks = np.array_split(order, bins)
    total = 0.0
    for chunk in chunks:
        if chunk.size == 0:
            continue
        total += (chunk.size / conf.size) * abs(corr[chunk].mean() - conf[chunk].mean())
    return float(total)


def maximum_calibration_error(
    confidence, correct, n_bins: int = 10  # noqa: ANN001
) -> float:
    """Worst per-bin calibration gap over equal-mass bins."""
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    bins = int(min(n_bins, conf.size))
    order = np.argsort(conf, kind="stable")
    worst = 0.0
    for chunk in np.array_split(order, bins):
        if chunk.size == 0:
            continue
        worst = max(worst, abs(corr[chunk].mean() - conf[chunk].mean()))
    return float(worst)


def brier_score(confidence, correct) -> float:  # noqa: ANN001
    """Mean squared error between confidence and correctness."""
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - corr) ** 2))


def negative_log_likelihood(confidence, correct, eps: float = 1e-12) -> float:  # noqa: ANN001
    """Mean binary NLL of correctness under the reported confidence."""
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    p = np.clip(conf, eps, 1.0 - eps)
    return float(-np.mean(corr * np.log(p) + (1.0 - corr) * np.log(1.0 - p)))


def risk_coverage_curve(confidence, correct) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """Risk (error rate) as a function of coverage, sorted by confidence.

    Returns:
        ``(coverage, risk)``, both length ``n``, where ``coverage[k] =
        (k+1)/n``. Empty arrays if nothing is finite.
    """
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(-conf, kind="stable")
    errors = 1.0 - corr[order]
    cumulative = np.cumsum(errors) / np.arange(1, conf.size + 1)
    coverage = np.arange(1, conf.size + 1) / conf.size
    return coverage, cumulative


def aurc(confidence, correct) -> float:  # noqa: ANN001
    """Area under the risk-coverage curve. Lower is better.

    Trapezoidal integration over coverage, which is the standard estimator
    (Geifman & El-Yaniv, 2017). Note that AURC depends on the base error rate, so
    it is comparable across methods only at similar accuracy -- the tables report
    accuracy next to it for that reason.
    """
    coverage, risk = risk_coverage_curve(confidence, correct)
    if coverage.size < 2:
        return float("nan")
    return float(np.trapezoid(risk, coverage))


def error_detection_auroc(confidence, correct) -> float:  # noqa: ANN001
    """AUROC for using confidence to detect *errors*.

    Computed with the Mann-Whitney U identity rather than by thresholding, so ties
    are handled exactly. Returns ``NaN`` when either class is empty -- with no
    errors there is nothing to detect and 0.5 would be a fabrication.
    """
    conf, corr = _clean(confidence, correct)
    if conf.size == 0:
        return float("nan")
    right = conf[corr > 0.5]
    wrong = conf[corr <= 0.5]
    if right.size == 0 or wrong.size == 0:
        return float("nan")
    ranks = _rankdata(np.concatenate([right, wrong]))
    rank_sum_right = ranks[: right.size].sum()
    u = rank_sum_right - right.size * (right.size + 1) / 2.0
    return float(u / (right.size * wrong.size))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. Avoids a scipy import in the hot path."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def summarise_calibration(confidence, correct, n_bins: int = 10) -> dict[str, float]:  # noqa: ANN001
    """Every calibration and selective-prediction metric, plus the item count."""
    conf, corr = _clean(confidence, correct)
    return {
        "ece": expected_calibration_error(conf, corr, n_bins),
        "ace": adaptive_calibration_error(conf, corr, n_bins),
        "mce": maximum_calibration_error(conf, corr, n_bins),
        "brier": brier_score(conf, corr),
        "nll": negative_log_likelihood(conf, corr),
        "aurc": aurc(conf, corr),
        "error_auroc": error_detection_auroc(conf, corr),
        "mean_confidence": float(conf.mean()) if conf.size else float("nan"),
        "n_calibration": float(conf.size),
    }
