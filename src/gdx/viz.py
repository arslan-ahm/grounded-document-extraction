"""Figures, drawn from the committed CSVs.

Two rules. First, **no backend is forced at import time**: doing so makes every
notebook plot render blank without an error. The backend is selected only when a
figure is written to a file, and only if the caller is not already inside an
IPython kernel. Second, every figure is a function of a committed CSV, so a
figure cannot show something the tables do not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIGURES = Path("results/figures")
TABLES = Path("results/tables")


def _pyplot():  # noqa: ANN202
    """Import pyplot, choosing Agg only when not already in a kernel."""
    import matplotlib

    if "ipykernel" not in sys.modules and matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, name: str, out_dir: Path) -> Path:  # noqa: ANN001
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    return path


def _read(name: str) -> pd.DataFrame | None:
    path = TABLES / f"{name}.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return None if frame.empty else frame


def figure_document(doc, out_dir: Path = FIGURES, name: str = "document") -> Path | None:
    """Draw one document's boxes with the ground-truth field spans highlighted.

    This is the figure that makes the benchmark inspectable: a reader can see the
    distractor amounts sitting next to the target, and see that the provenance
    truth points at a specific region rather than at a string.
    """
    plt = _pyplot()
    from gdx.data.schema import FIELDS

    n_pages = doc.n_pages
    fig, axes = plt.subplots(1, n_pages, figsize=(4.2 * n_pages, 5.6), squeeze=False)
    colours = plt.get_cmap("tab10")
    highlight = {}
    for i, field_name in enumerate(FIELDS):
        truth = doc.fields[field_name]
        for span in truth.all_spans:
            for idx in range(span[0], span[1] + 1):
                highlight[idx] = (field_name, colours(i % 10))

    for page in range(n_pages):
        ax = axes[0][page]
        for idx, tok in enumerate(doc.tokens):
            if tok.page != page:
                continue
            x0, y0, x1, y1 = tok.box
            info = highlight.get(idx)
            colour = info[1] if info else "0.75"
            ax.add_patch(
                plt.Rectangle(
                    (x0, 1.0 - y1), x1 - x0, y1 - y0,
                    facecolor=colour, edgecolor="none", alpha=0.85 if info else 0.35,
                )
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"page {page + 1}/{n_pages}", fontsize=9)
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=colours(i % 10), label=f)
        for i, f in enumerate(FIELDS)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(
        f"document {doc.doc_id}: {len(doc.tokens)} tokens, grey = context, "
        "coloured = ground-truth provenance",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path = _save(fig, name, out_dir)
    plt.close(fig)
    return path


def figure_hallucination(out_dir: Path = FIGURES) -> Path | None:
    """Hallucination rate and coverage per arm, side by side.

    Both bars matter together: a zero hallucination rate at zero coverage is not
    a result, so the figure refuses to show one without the other.
    """
    frame = _read("method_comparison")
    if frame is None:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(frame))
    ax.bar(x - 0.2, frame["hallucination_rate"].fillna(0.0), 0.38,
           label="hallucination rate", color="#c0392b")
    ax.bar(x + 0.2, frame["coverage"], 0.38, label="coverage", color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["arm"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Emitted values that appear nowhere in the document, and how often each arm emits")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    path = _save(fig, "hallucination_and_coverage", out_dir)
    plt.close(fig)
    return path


def figure_risk_coverage(out_dir: Path = FIGURES, seed: int = 0) -> Path | None:
    """Risk-coverage curves from the per-item CSVs."""
    from gdx.metrics.calibration import risk_coverage_curve
    from gdx.pipelines.analysis import load_per_item

    try:
        frame = load_per_item()
    except FileNotFoundError:
        return None
    frame = frame[(frame["seed"] == seed) & frame["emitted"]]
    if frame.empty:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for arm, group in frame.groupby("arm"):
        coverage, risk = risk_coverage_curve(
            group["confidence"].to_numpy(dtype=np.float64),
            group["canonical_correct"].to_numpy(dtype=np.float64),
        )
        if coverage.size:
            ax.plot(coverage, risk, label=arm, linewidth=1.6)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate)")
    ax.set_title(f"Risk-coverage on emitted values, seed {seed}")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    path = _save(fig, "risk_coverage", out_dir)
    plt.close(fig)
    return path


def figure_normalisation_cost(out_dir: Path = FIGURES) -> Path | None:
    """The structural limitation, drawn: strict accuracy by subset."""
    frame = _read("normalisation_cost")
    if frame is None:
        return None
    plt = _pyplot()
    pivot = frame.pivot(index="arm", columns="subset", values="strict_accuracy")
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    pivot.plot(kind="bar", ax=ax, color=["#2c7fb8", "#e67e22"], width=0.75)
    ax.set_ylabel("strict accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict accuracy on verbatim fields vs fields needing normalisation")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    path = _save(fig, "normalisation_cost", out_dir)
    plt.close(fig)
    return path


def figure_efficiency(out_dir: Path = FIGURES) -> Path | None:
    """Decode latency against value length, with the fitted exponents."""
    frame = _read("decode_scaling")
    if frame is None:
        return None
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for head, group in frame.groupby("head"):
        exponent = float(group["fitted_exponent"].iloc[0])
        ax.plot(group["dec_max_len"], group["latency_ms"], marker="o",
                label=f"{head} (fitted exponent {exponent:.2f})")
    ax.set_xlabel("maximum value length (characters)")
    ax.set_ylabel("decode latency, batch 8 (ms)")
    ax.set_title("Selection decodes in constant time; generation does not")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    path = _save(fig, "decode_scaling", out_dir)
    plt.close(fig)
    return path


def figure_seed_variance(out_dir: Path = FIGURES) -> Path | None:
    """Per-arm metric spread across seeds, against the noise scale."""
    frame = _read("seed_variance")
    if frame is None:
        return None
    plt = _pyplot()
    metrics = ["strict_accuracy", "canonical_accuracy", "coverage"]
    subset = frame[frame["metric"].isin(metrics)]
    if subset.empty:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 3.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=True):
        rows = subset[subset["metric"] == metric]
        ax.errorbar(
            np.arange(len(rows)), rows["mean"],
            yerr=rows["noise_scale"].fillna(0.0) / 2.0,
            fmt="o", capsize=3, color="#2c7fb8",
        )
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels(rows["arm"], rotation=45, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Seed means with error bars of half the run-to-run noise scale", fontsize=9)
    fig.tight_layout()
    path = _save(fig, "seed_variance", out_dir)
    plt.close(fig)
    return path


def figure_training_curves(out_dir: Path = FIGURES, seed: int = 0) -> Path | None:
    """Train and validation loss for both heads, from ``history.jsonl``."""
    from gdx.utils.logging import JsonlLogger

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    drawn = False
    for head in ("span", "generative"):
        records = JsonlLogger.read(Path(f"results/runs/seed{seed}/{head}/history.jsonl"))
        if not records:
            continue
        epochs = [r["epoch"] for r in records]
        ax.plot(epochs, [r["train_loss"] for r in records], label=f"{head} train")
        ax.plot(epochs, [r["val_loss"] for r in records], "--", label=f"{head} val")
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Training curves, seed {seed} (the two heads' losses are not comparable)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    path = _save(fig, "training_curves", out_dir)
    plt.close(fig)
    return path


def make_all(out_dir: Path = FIGURES, seed: int = 0) -> dict[str, Path]:
    """Draw every figure whose source CSV exists. Missing sources are skipped."""
    from gdx.config import DataConfig
    from gdx.data.generator import generate_document

    out: dict[str, Path] = {}
    doc = generate_document(0, DataConfig(), seed=0)
    for name, fn in (
        ("document", lambda: figure_document(doc, out_dir)),
        ("hallucination_and_coverage", lambda: figure_hallucination(out_dir)),
        ("risk_coverage", lambda: figure_risk_coverage(out_dir, seed)),
        ("normalisation_cost", lambda: figure_normalisation_cost(out_dir)),
        ("decode_scaling", lambda: figure_efficiency(out_dir)),
        ("seed_variance", lambda: figure_seed_variance(out_dir)),
        ("training_curves", lambda: figure_training_curves(out_dir, seed)),
    ):
        path = fn()
        if path is not None:
            out[name] = path
    return out
