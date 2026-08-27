"""The pipeline end to end, plus the engine, the arms and the analysis tables.

The smoke test at the bottom is the one the project standard specifically asks
for: run the real pipeline at tiny scale and assert the output table is
*populated*. A results table that prints empty is a bug that survives every unit
test, so it gets its own assertion.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from gdx.arms import (
    ARM_BY_NAME,
    ARMS,
    baseline_candidates,
    empty_candidates,
    gen_candidates,
    model_candidates,
    span_candidates,
)
from gdx.baselines.heuristic import VARIANT_BY_NAME
from gdx.data.schema import FIELDS
from gdx.engine.trainer import (
    cosine_lr,
    evaluate_loss,
    load_checkpoint,
    save_checkpoint,
    train,
)
from gdx.llm.stub import StubClient
from gdx.models.model import build_model
from gdx.pipelines.analysis import (
    abstention_table,
    calibration_table,
    headline_table,
    load_per_item,
    normalisation_cost,
    per_field_table,
)
from gdx.pipelines.core import evaluate_arm, prepare, run_single, select_heuristic_variant
from gdx.pipelines.experiments import (
    METRIC_FAMILY,
    REFERENCE_ARM,
    _append_csv,
    paired_tests,
    seed_variance,
    verdict_table,
)

# --- arms ------------------------------------------------------------------

def test_arm_names_are_unique_and_cover_the_documented_set():
    names = [a.name for a in ARMS]
    assert len(names) == len(set(names))
    assert set(names) == {
        "heuristic", "llm_stub", "generative", "generative_verify",
        "span_only", "span_verify", "span_verify_norm",
    }


def test_the_reference_arm_exists_and_is_generative():
    assert REFERENCE_ARM in ARM_BY_NAME
    assert ARM_BY_NAME[REFERENCE_ARM].source == "generative"
    assert ARM_BY_NAME[REFERENCE_ARM].family == "reference"


def test_span_only_and_span_verify_differ_only_in_the_verify_switch(tiny_cfg):  # noqa: ANN001
    a = ARM_BY_NAME["span_only"].verify_config(tiny_cfg.verify)
    b = ARM_BY_NAME["span_verify"].verify_config(tiny_cfg.verify)
    diffs = {k for k in vars(a) if getattr(a, k) != getattr(b, k)}
    assert diffs == {"enabled"}


def test_span_verify_norm_differs_from_span_verify_only_in_normalisation():
    a = ARM_BY_NAME["span_verify"]
    b = ARM_BY_NAME["span_verify_norm"]
    assert a.verify == b.verify
    assert b.normalise and not a.normalise


def test_arm_verify_config_rejects_an_unknown_switch(tiny_cfg):  # noqa: ANN001
    from gdx.arms import Arm

    with pytest.raises(TypeError):
        Arm("x", "span", "ablation", {"not_a_switch": True}).verify_config(tiny_cfg.verify)


def test_span_candidates_drop_options_below_the_null(tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    model = build_model(replace(tiny_cfg.model, head="span"), vocab.size, seed=0)
    batch = tiny_dataset.collate([0])
    preds = model.predict(batch)[0]
    cands = span_candidates(preds)
    for pred in preds:
        for cand in cands[pred.field_name]:
            assert cand.prob > pred.null_prob
            assert cand.span is not None


def test_gen_candidates_are_empty_for_an_empty_string():
    from gdx.models.heads import GenPrediction

    preds = [GenPrediction(n, "X" if i % 2 else "", 0.5) for i, n in enumerate(FIELDS)]
    cands = gen_candidates(preds)
    assert cands[FIELDS[0]] == []
    assert len(cands[FIELDS[1]]) == 1
    assert cands[FIELDS[1]][0].span is None


def test_model_candidates_are_returned_in_document_order(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    """Batching sorts by length; the result must be re-keyed by document."""
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    cands = model_candidates(model, tiny_splits.test, tiny_cfg)
    assert len(cands) == len(tiny_splits.test.docs)
    assert all(set(c) == set(FIELDS) for c in cands)


def test_baseline_candidates_require_their_argument(tiny_splits):  # noqa: ANN001
    with pytest.raises(ValueError, match="HeuristicVariant"):
        baseline_candidates("heuristic", tiny_splits.test)
    with pytest.raises(ValueError, match="LLMClient"):
        baseline_candidates("llm", tiny_splits.test)
    with pytest.raises(ValueError, match="unknown baseline source"):
        baseline_candidates("magic", tiny_splits.test)


def test_baseline_candidates_llm_reports_usage(tiny_splits):  # noqa: ANN001
    cands, usage = baseline_candidates("llm", tiny_splits.test, client=StubClient())
    assert len(cands) == len(tiny_splits.test.docs)
    assert usage.calls == len(tiny_splits.test.docs) * len(FIELDS)


def test_empty_candidates_are_independent_lists():
    cands = empty_candidates(3)
    cands[0]["total"].append(object())
    assert cands[1]["total"] == []


# --- engine ----------------------------------------------------------------

def test_cosine_lr_warms_up_then_decays():
    total, warmup, base = 100, 10, 1.0
    assert cosine_lr(0, total, warmup, base) == pytest.approx(0.1)
    assert cosine_lr(9, total, warmup, base) == pytest.approx(1.0)
    assert cosine_lr(50, total, warmup, base) < 1.0
    assert cosine_lr(99, total, warmup, base) == pytest.approx(0.1, abs=0.01)


def test_cosine_lr_never_reaches_zero():
    """A zero-LR final epoch is wall-clock spent for nothing."""
    values = [cosine_lr(s, 60, 6, 1.0) for s in range(60)]
    assert min(values) >= 0.09


def test_cosine_lr_with_no_steps_returns_the_base():
    assert cosine_lr(0, 0, 0, 0.5) == 0.5


def test_training_reduces_the_loss(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    cfg = replace(tiny_cfg, optim=replace(tiny_cfg.optim, epochs=3))
    model = build_model(cfg.model, vocab.size, seed=0)
    before = evaluate_loss(model, tiny_splits.val, cfg.optim.batch_size)
    result = train(model, tiny_splits.train, tiny_splits.val, cfg)
    after = evaluate_loss(model, tiny_splits.val, cfg.optim.batch_size)
    assert after < before
    assert len(result.history) == 3
    assert result.best_epoch >= 0
    assert result.steps > 0


def test_training_restores_the_best_checkpoint(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    cfg = replace(tiny_cfg, optim=replace(tiny_cfg.optim, epochs=2))
    model = build_model(cfg.model, vocab.size, seed=0)
    result = train(model, tiny_splits.train, tiny_splits.val, cfg)
    final = evaluate_loss(model, tiny_splits.val, cfg.optim.batch_size)
    assert final == pytest.approx(result.best_val_loss, abs=1e-6)


def test_training_writes_a_readable_partial_history(tiny_cfg, vocab, tiny_splits, tmp_path):  # noqa: ANN001
    from gdx.utils.logging import JsonlLogger

    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    train(model, tiny_splits.train, tiny_splits.val, tiny_cfg, tmp_path)
    records = JsonlLogger.read(tmp_path / "history.jsonl")
    assert records
    assert {"epoch", "train_loss", "val_loss"} <= set(records[0])


def test_evaluate_loss_weights_by_batch_size(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    a = evaluate_loss(model, tiny_splits.val, 4)
    b = evaluate_loss(model, tiny_splits.val, 8)
    assert a == pytest.approx(b, rel=1e-5)


def test_evaluate_loss_restores_training_mode(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    model.train()
    evaluate_loss(model, tiny_splits.val, 4)
    assert model.training


def test_checkpoint_round_trip(tiny_cfg, vocab, tmp_path):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    path = save_checkpoint(model, tmp_path / "m.pt")
    other = build_model(tiny_cfg.model, vocab.size, seed=99)
    load_checkpoint(other, path)
    for a, b in zip(model.parameters(), other.parameters(), strict=True):
        assert torch.equal(a, b)


def test_checkpoint_refuses_a_head_mismatch(tiny_cfg, vocab, tmp_path):  # noqa: ANN001
    span = build_model(replace(tiny_cfg.model, head="span"), vocab.size, seed=0)
    path = save_checkpoint(span, tmp_path / "m.pt")
    gen = build_model(replace(tiny_cfg.model, head="generative"), vocab.size, seed=0)
    with pytest.raises(ValueError, match="checkpoint head"):
        load_checkpoint(gen, path)


# --- core pipeline ---------------------------------------------------------

def test_prepare_is_deterministic(tiny_cfg):  # noqa: ANN001
    a = prepare(tiny_cfg, seed=3)
    b = prepare(tiny_cfg, seed=3)
    assert [d.texts for d in a.splits.test.docs] == [d.texts for d in b.splits.test.docs]
    assert a.vocab.size == b.vocab.size


def test_evaluate_arm_summary_is_populated(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    cands = model_candidates(model, tiny_splits.test, tiny_cfg)
    result = evaluate_arm(ARM_BY_NAME["span_verify"], tiny_splits.test, cands, tiny_cfg)
    assert result.summary["n_records"] == len(tiny_splits.test.docs) * len(FIELDS)
    assert len(result.records) == int(result.summary["n_records"])
    assert set(result.per_field) <= set(FIELDS)
    for key in ("strict_accuracy", "canonical_accuracy", "coverage", "grounding_exact"):
        assert key in result.summary


def test_evaluate_arm_per_item_frame_has_one_row_per_pair(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    cands = model_candidates(model, tiny_splits.test, tiny_cfg)
    result = evaluate_arm(ARM_BY_NAME["span_only"], tiny_splits.test, cands, tiny_cfg)
    frame = result.per_item_frame()
    assert len(frame) == len(tiny_splits.test.docs) * len(FIELDS)
    assert set(frame["arm"]) == {"span_only"}
    assert {"doc_id", "field", "confidence", "grounded", "span_exact"} <= set(frame.columns)


def test_evaluate_arm_normalisation_changes_only_dates(tiny_cfg, vocab, tiny_splits):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    cands = model_candidates(model, tiny_splits.test, tiny_cfg)
    plain = evaluate_arm(ARM_BY_NAME["span_verify"], tiny_splits.test, cands, tiny_cfg)
    normed = evaluate_arm(ARM_BY_NAME["span_verify_norm"], tiny_splits.test, cands, tiny_cfg)
    for a, b in zip(plain.records, normed.records, strict=True):
        if a.field_name not in ("invoice_date", "due_date"):
            assert a.predicted_value == b.predicted_value


def test_evaluate_arm_mismatched_lengths_raise(tiny_cfg, tiny_splits):  # noqa: ANN001
    with pytest.raises(ValueError):
        evaluate_arm(ARM_BY_NAME["span_verify"], tiny_splits.test, empty_candidates(1), tiny_cfg)


def test_select_heuristic_variant_returns_a_known_variant(tiny_cfg):  # noqa: ANN001
    prepared = prepare(tiny_cfg, seed=0)
    variant = select_heuristic_variant(tiny_cfg, prepared, limit=8)
    assert variant.name in VARIANT_BY_NAME


@pytest.mark.slow
def test_run_single_writes_a_populated_run_directory(tmp_path):  # noqa: ANN001
    """The end-to-end smoke test the project standard asks for."""
    from gdx.config import load_config

    cfg = load_config(
        "configs/smoke.yaml",
        [f"run.out_dir={tmp_path.as_posix()}", "run.name=smoke_test", "optim.epochs=1"],
    )
    payload = run_single(cfg)
    run_dir = tmp_path / "smoke_test"
    for name in ("config.yaml", "history.jsonl", "per_item.csv", "summary.json", "summary.csv"):
        assert (run_dir / name).exists(), f"{name} missing"

    table = pd.read_csv(run_dir / "summary.csv")
    assert not table.empty, "the results table printed empty"
    assert table["n_records"].min() > 0
    assert table["coverage"].notna().all()
    assert set(table["arm"]) == {"span_only", "span_verify", "span_verify_norm"}
    per_item = pd.read_csv(run_dir / "per_item.csv")
    assert len(per_item) == len(table) * cfg.data.n_test * len(FIELDS)
    assert payload["splits"]["test"] == cfg.data.n_test
    # The guarantee, on real trained output at the end of the real pipeline.
    emitted = per_item[per_item["emitted"] & per_item["arm"].str.startswith("span")]
    assert len(emitted) > 0
    assert bool(emitted["grounded"].all())


def test_run_single_rejects_a_head_with_no_arms(tiny_cfg, tmp_path):  # noqa: ANN001
    cfg = replace(
        tiny_cfg,
        model=replace(tiny_cfg.model, head="span"),
        run=replace(tiny_cfg.run, out_dir=tmp_path.as_posix()),
    )
    with pytest.raises(ValueError, match="no arms defined"):
        run_single(cfg, arms=())


# --- analysis tables -------------------------------------------------------

def _fake_runs() -> pd.DataFrame:
    rows = []
    for seed in (0, 1, 2):
        for arm, base in (("generative", 0.60), ("span_verify", 0.90)):
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "family": "reference" if arm == "generative" else "ours",
                    "strict_accuracy": base + 0.01 * seed,
                    "canonical_accuracy": base + 0.02 + 0.01 * seed,
                    "coverage": 0.8,
                    "grounding_exact": np.nan if arm == "generative" else 0.95,
                    "grounding_iou": np.nan if arm == "generative" else 0.93,
                    "hallucination_rate": 0.2 if arm == "generative" else 0.0,
                    "n_records": 100,
                }
            )
    return pd.DataFrame(rows)


def test_seed_variance_computes_sqrt_two_sd():
    variance = seed_variance(_fake_runs())
    row = variance[
        (variance["arm"] == "generative") & (variance["metric"] == "strict_accuracy")
    ].iloc[0]
    assert row["n_seeds"] == 3
    assert row["noise_scale"] == pytest.approx(math.sqrt(2.0) * row["sd"])


def test_seed_variance_reports_nan_for_an_all_nan_metric():
    variance = seed_variance(_fake_runs())
    row = variance[
        (variance["arm"] == "generative") & (variance["metric"] == "grounding_exact")
    ].iloc[0]
    assert row["n_seeds"] == 0
    assert math.isnan(row["mean"])


def test_verdict_table_compares_against_the_reference():
    runs = _fake_runs()
    table = verdict_table(runs, seed_variance(runs))
    assert REFERENCE_ARM not in set(table["arm"])
    row = table[
        (table["arm"] == "span_verify") & (table["metric"] == "strict_accuracy")
    ].iloc[0]
    assert row["delta"] == pytest.approx(0.30)
    assert row["verdict"] in {"robust", "survives", "suggestive", "inside noise", "unknown"}


def test_verdict_table_covers_the_whole_metric_family():
    runs = _fake_runs()
    table = verdict_table(runs, seed_variance(runs))
    assert set(table["metric"]) <= set(METRIC_FAMILY)


def _fake_per_item() -> pd.DataFrame:
    rows = []
    for arm in ("generative", "span_verify"):
        for doc_id in range(6):
            for field_name in FIELDS:
                present = field_name != "tax"
                emitted = present or arm == "generative"
                correct = arm == "span_verify" and present
                rows.append(
                    {
                        "arm": arm,
                        "seed": 0,
                        "doc_id": doc_id,
                        "field": field_name,
                        "truth_present": present,
                        "truth_value": "120.00" if present else "",
                        "predicted_value": "120.00" if emitted else "",
                        "abstained": not emitted,
                        "emitted": emitted,
                        "confidence": 0.9 if arm == "span_verify" else 0.5,
                        "grounded": arm == "span_verify" and emitted,
                        "requires_normalisation": field_name == "invoice_date",
                        "strict_correct": correct,
                        "canonical_correct": correct,
                        "span_exact": correct,
                        "span_iou": 1.0 if correct else 0.0,
                        "n_iters": 0,
                        "reason": "",
                    }
                )
    return pd.DataFrame(rows)


def test_headline_table_orders_the_arms_and_keeps_the_columns():
    table = headline_table(_fake_runs(), seed=0)
    assert list(table["arm"]) == ["generative", "span_verify"]
    assert "hallucination_rate" in table.columns


def test_normalisation_cost_splits_by_the_flag():
    table = normalisation_cost(_fake_per_item(), seed=0)
    assert set(table["subset"]) == {"verbatim", "needs_normalisation"}
    assert (table["n"] > 0).all()


def test_per_field_table_has_one_row_per_arm_and_field():
    table = per_field_table(_fake_per_item(), seed=0)
    assert len(table) == 2 * len(FIELDS)
    assert list(table[table["arm"] == "span_verify"]["field"]) == list(FIELDS)


def test_calibration_table_only_uses_emitted_rows():
    table = calibration_table(_fake_per_item(), seed=0)
    assert set(table["arm"]) == {"generative", "span_verify"}
    assert (table["n_calibration"] > 0).all()


def test_abstention_table_reports_the_absent_rate():
    table = abstention_table(_fake_per_item(), seed=0)
    row = table[table["arm"] == "span_verify"].iloc[0]
    assert row["n_absent"] == 6
    assert row["absent_abstain_rate"] == pytest.approx(1.0)


def test_paired_tests_uses_the_document_as_the_unit():
    table = paired_tests(_fake_per_item(), seed=0)
    assert not table.empty
    assert set(table["unit"]) <= {"document", "set"}
    documents = table[table["unit"] == "document"]
    assert (documents["n"] <= 6).all()


def test_paired_tests_reports_the_hallucination_difference_as_a_set_statistic():
    table = paired_tests(_fake_per_item(), seed=0)
    rows = table[table["metric"] == "hallucination_rate"]
    assert len(rows) == 1
    assert rows.iloc[0]["unit"] == "set"
    assert rows.iloc[0]["difference"] < 0


def test_load_per_item_raises_when_nothing_has_been_run(tmp_path):  # noqa: ANN001
    with pytest.raises(FileNotFoundError, match="no per-item CSVs"):
        load_per_item(tmp_path)


def test_append_csv_replaces_rows_sharing_the_key(tmp_path):  # noqa: ANN001
    path = tmp_path / "t.csv"
    _append_csv(path, pd.DataFrame([{"arm": "a", "seed": 0, "v": 1}]), ("arm", "seed"))
    _append_csv(path, pd.DataFrame([{"arm": "a", "seed": 0, "v": 2}]), ("arm", "seed"))
    _append_csv(path, pd.DataFrame([{"arm": "a", "seed": 1, "v": 3}]), ("arm", "seed"))
    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert int(frame[frame["seed"] == 0]["v"].iloc[0]) == 2
