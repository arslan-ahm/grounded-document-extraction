"""Core pipeline: prepare data, train a head, evaluate an arm, write a run dir.

Everything the scripts do is assembled from these four functions, so the tests can
reach the whole pipeline without going through ``argparse``.

One convention worth stating: **calibration is measured on emitted values only.**
An abstention carries a confidence too -- the null probability -- but that number
is a confidence in *absence*, not in a value, and pooling the two would produce a
reliability diagram about two different questions at once. Coverage,
``absent_abstain_rate`` and ``present_emit_rate`` describe the abstention
behaviour instead, and the key names say which set each metric came from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from gdx.arms import Arm, baseline_candidates, model_candidates
from gdx.baselines.heuristic import VARIANT_BY_NAME, VARIANTS, HeuristicVariant
from gdx.config import Config
from gdx.data.dataset import DocumentDataset, Splits, build_splits
from gdx.data.featurise import Vocabulary, build_vocabulary
from gdx.extract import Extraction, normalise_extraction
from gdx.llm.stub import build_client
from gdx.metrics.calibration import summarise_calibration
from gdx.metrics.fields import FieldRecord, build_records, summarise, summarise_per_field
from gdx.metrics.grounding import span_exact, span_iou, summarise_grounding
from gdx.models.model import GDXModel, build_model
from gdx.utils.logging import get_logger, write_json
from gdx.utils.seed import seed_everything
from gdx.verify import VerificationTrace, verify_document

LOG = get_logger(__name__)


@dataclass
class Prepared:
    """Vocabulary and the three splits for one seed."""

    vocab: Vocabulary
    splits: Splits
    seed: int


@dataclass
class EvalResult:
    """One arm's evaluation on one split."""

    arm: str
    records: list[FieldRecord] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    per_field: dict[str, dict[str, float]] = field(default_factory=dict)
    traces: list[VerificationTrace] = field(default_factory=list)
    seconds: float = 0.0

    def to_row(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        row: dict[str, Any] = {"arm": self.arm}
        row.update(self.summary)
        row["eval_seconds"] = self.seconds
        row.update(extra or {})
        return row

    def per_item_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([r.to_dict() for r in self.records])
        frame.insert(0, "arm", self.arm)
        return frame


def prepare(cfg: Config, seed: int | None = None) -> Prepared:
    """Seed everything, build the vocabulary and generate the splits."""
    use_seed = cfg.run.seed if seed is None else seed
    seed_everything(use_seed, cfg.run.n_threads)
    vocab = build_vocabulary(cfg.model.vocab_size)
    splits = build_splits(cfg.data, cfg.model, vocab, seed=use_seed)
    return Prepared(vocab=vocab, splits=splits, seed=use_seed)


def train_head(
    cfg: Config, prepared: Prepared, head: str, run_dir: Path | None = None
) -> tuple[GDXModel, dict[str, Any]]:
    """Train one head under ``cfg``, returning the model and its history summary.

    The model is built *after* re-seeding on ``prepared.seed`` so that both heads
    see the same encoder initialisation for a given seed. Without that, part of
    the span-versus-generation gap would be an initialisation difference.
    """
    from dataclasses import replace

    from gdx.engine.trainer import train

    model_cfg = replace(cfg.model, head=head)
    seed_everything(prepared.seed, cfg.run.n_threads)
    model = build_model(model_cfg, prepared.vocab.size, seed=prepared.seed)
    run_cfg = replace(cfg, model=model_cfg)
    LOG.info("training head=%s seed=%d params=%d", head, prepared.seed,
             sum(p.numel() for p in model.parameters()))
    result = train(model, prepared.splits.train, prepared.splits.val, run_cfg, run_dir)
    return model, result.to_dict()


def evaluate_arm(
    arm: Arm,
    data: DocumentDataset,
    candidates: list[dict[str, list]],
    cfg: Config,
) -> EvalResult:
    """Run the verification loop for ``arm`` and compute every metric.

    Args:
        arm: The arm, supplying its verification overrides and normalisation flag.
        data: The split being evaluated.
        candidates: Per-document candidate sets, in ``data.docs`` order.
        cfg: Full config; ``cfg.verify`` is the base the arm overrides.

    Returns:
        An :class:`EvalResult`.
    """
    started = time.perf_counter()
    verify_cfg = arm.verify_config(cfg.verify)
    records: list[FieldRecord] = []
    traces: list[VerificationTrace] = []
    all_extractions: list[dict[str, Extraction]] = []

    for doc, cands in zip(data.docs, candidates, strict=True):
        extractions, trace = verify_document(doc, cands, verify_cfg)
        if arm.normalise:
            extractions = {k: normalise_extraction(v) for k, v in extractions.items()}
        traces.append(trace)
        all_extractions.append(extractions)
        records.extend(build_records(doc, extractions, span_exact, span_iou))

    summary = summarise(records)
    summary.update(summarise_grounding(data.docs, all_extractions))
    emitted = [r for r in records if r.emitted]
    summary.update(
        summarise_calibration(
            [r.confidence for r in emitted], [float(r.canonical_correct) for r in emitted]
        )
    )
    n_traces = max(1, len(traces))
    summary["verify_iters_per_doc"] = sum(t.n_iters for t in traces) / n_traces
    summary["verify_reselections_per_doc"] = sum(t.reselections for t in traces) / n_traces
    summary["verify_arith_checked_frac"] = (
        sum(1 for t in traces if t.arithmetic_checked) / n_traces
    )
    checked = [t for t in traces if t.arithmetic_checked]
    summary["verify_arith_ok_frac"] = (
        sum(1 for t in checked if t.arithmetic_final_ok) / len(checked)
        if checked
        else float("nan")
    )
    return EvalResult(
        arm=arm.name,
        records=records,
        summary=summary,
        per_field=summarise_per_field(records),
        traces=traces,
        seconds=time.perf_counter() - started,
    )


def select_heuristic_variant(cfg: Config, prepared: Prepared, limit: int = 150) -> HeuristicVariant:
    """Choose the rule baseline's geometry on **validation**, never on test.

    Args:
        cfg: Config, for the verification base.
        prepared: The prepared splits.
        limit: Validation documents used for the sweep. Capped because the sweep
            is six configurations and the baseline costs ~10 ms per document;
            the cap is documented rather than silent.

    Returns:
        The winning :class:`HeuristicVariant` by canonical accuracy.
    """
    from gdx.arms import ARM_BY_NAME

    arm = ARM_BY_NAME["heuristic"]
    subset = DocumentDataset(
        prepared.splits.val.docs[:limit], prepared.vocab, cfg.model.dec_max_len
    )
    best: tuple[float, HeuristicVariant] | None = None
    for variant in VARIANTS:
        cands, _ = baseline_candidates("heuristic", subset, variant=variant)
        result = evaluate_arm(arm, subset, cands, cfg)
        score = result.summary.get("canonical_accuracy", float("nan"))
        LOG.info("heuristic variant %-18s val canonical_accuracy %.4f", variant.name, score)
        if score == score and (best is None or score > best[0]):
            best = (score, variant)
    chosen = best[1] if best is not None else VARIANT_BY_NAME["row_first"]
    LOG.info("heuristic variant chosen on validation: %s", chosen.name)
    return chosen


def run_single(cfg: Config, arms: tuple[Arm, ...] | None = None) -> dict[str, Any]:
    """Train the configured head, evaluate its arms, and write the run directory.

    This is what ``scripts/train.py`` calls, and what the end-to-end smoke test
    asserts on. It writes ``config.yaml``, ``history.jsonl``, ``per_item.csv`` and
    ``summary.json`` into ``results/runs/<name>/``.
    """
    from gdx.arms import ARMS

    # `arms is None` rather than `not arms`: an explicitly empty tuple is a
    # caller error and must raise, not silently fall back to the defaults.
    if arms is None:
        arms = tuple(a for a in ARMS if a.source == cfg.model.head)
    if not arms:
        raise ValueError(f"no arms defined for head {cfg.model.head!r}")
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    prepared = prepare(cfg)
    model, history = train_head(cfg, prepared, cfg.model.head, run_dir)
    test = prepared.splits.test
    cands = model_candidates(model, test, cfg)

    rows = []
    frames = []
    for arm in arms:
        result = evaluate_arm(arm, test, cands, cfg)
        rows.append(result.to_row({"head": cfg.model.head, "seed": prepared.seed}))
        if cfg.run.save_per_item:
            frames.append(result.per_item_frame())
        LOG.info(
            "%-18s strict %.4f canonical %.4f coverage %.4f hallucination %s",
            arm.name,
            result.summary.get("strict_accuracy", float("nan")),
            result.summary.get("canonical_accuracy", float("nan")),
            result.summary.get("coverage", float("nan")),
            f"{result.summary.get('hallucination_rate', float('nan')):.4f}",
        )

    table = pd.DataFrame(rows)
    table.to_csv(run_dir / "summary.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(run_dir / "per_item.csv", index=False)
    payload = {
        "config": cfg.to_dict(),
        "history": history,
        "splits": prepared.splits.sizes,
        "arms": rows,
    }
    write_json(run_dir / "summary.json", payload)
    return payload


def build_llm_client(cfg: Config):  # noqa: ANN201
    """The LLM arm's client. Defaults to the offline stub, which needs no key."""
    return build_client(cfg.baseline.llm_backend, cfg.baseline.llm_model,
                        cfg.baseline.llm_temperature)
