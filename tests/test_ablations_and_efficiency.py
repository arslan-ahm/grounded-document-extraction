"""Ablation bookkeeping and the efficiency measurement machinery.

The ablation tests care about *attribution*: each switch must flip exactly one
config field, the inference-time family must reuse one checkpoint so its deltas
carry no initialisation noise, and every delta must be divided by a noise scale
before it is interpreted. The efficiency tests care about the measurement
protocol -- warm-up, repeats, and the fitted exponent -- because an under-warmed
timing table is worse than no timing table.
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gdx.config import ModelConfig, VerifyConfig
from gdx.pipelines.ablations import (
    MODEL_ABLATIONS,
    VERIFY_ABLATIONS,
    run_model_ablations,
    run_verify_ablations,
    summarise_ablations,
    train_shared_span,
)
from gdx.pipelines.efficiency import baseline_cost, decode_scaling, head_cost, torch_thread_note

# --- ablation definitions --------------------------------------------------

def test_verify_ablations_start_with_full_and_are_unique():
    names = [n for n, _ in VERIFY_ABLATIONS]
    assert names[0] == "full"
    assert len(names) == len(set(names))
    assert VERIFY_ABLATIONS[0][1] == {}


def test_model_ablations_start_with_full_and_are_unique():
    names = [n for n, _ in MODEL_ABLATIONS]
    assert names[0] == "full"
    assert len(names) == len(set(names))


def test_every_verify_ablation_flips_exactly_one_field():
    """A variant that changed two things would attribute nothing."""
    valid = {f.name for f in dataclass_fields(VerifyConfig)}
    for name, overrides in VERIFY_ABLATIONS:
        assert set(overrides) <= valid, f"{name} names a non-existent switch"
        assert len(overrides) <= 1, f"{name} changes {len(overrides)} things"


def test_every_model_ablation_flips_exactly_one_field():
    valid = {f.name for f in dataclass_fields(ModelConfig)}
    for name, overrides in MODEL_ABLATIONS:
        assert set(overrides) <= valid, f"{name} names a non-existent switch"
        assert len(overrides) <= 1, f"{name} changes {len(overrides)} things"


def test_verify_ablations_cover_every_mechanism():
    """Verification, type check, arithmetic and abstention each get a switch."""
    switches = {k for _, o in VERIFY_ABLATIONS for k in o}
    assert {"enabled", "type_check", "arithmetic", "abstain"} <= switches


def test_model_ablations_cover_both_layout_signals():
    switches = {k for _, o in MODEL_ABLATIONS for k in o}
    assert switches == {"use_2d_pos", "use_spatial_bias"}


# --- ablation execution ----------------------------------------------------

@pytest.fixture
def ablation_cfg(tiny_cfg):  # noqa: ANN001, ANN201
    return replace(tiny_cfg, run=replace(tiny_cfg.run, name="abl_test"))


def test_verify_ablations_share_one_checkpoint(ablation_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    """The whole point of the inference family: identical candidates every row."""
    from gdx.pipelines import ablations as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    shared = train_shared_span(ablation_cfg, 0)
    frame = run_verify_ablations(ablation_cfg, 0, shared=shared)
    assert len(frame) == len(VERIFY_ABLATIONS)
    assert set(frame["kind"]) == {"inference"}
    # One training run, so every row reports the same training time.
    assert frame["train_seconds"].nunique() == 1


def test_verify_ablation_no_verification_raises_coverage(ablation_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    """Turning the loop off can only emit more, never fewer, values."""
    from gdx.pipelines import ablations as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    shared = train_shared_span(ablation_cfg, 0)
    frame = run_verify_ablations(ablation_cfg, 0, shared=shared).set_index("ablation")
    assert frame.loc["no_verification", "coverage"] >= frame.loc["full", "coverage"]


def test_verify_ablation_hallucination_stays_zero_throughout(ablation_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    """Every inference switch is downstream of selection, so (P1) is untouched."""
    from gdx.pipelines import ablations as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    shared = train_shared_span(ablation_cfg, 0)
    frame = run_verify_ablations(ablation_cfg, 0, shared=shared)
    assert (frame["hallucination_rate"].fillna(0.0) == 0.0).all()


def test_model_ablations_reuse_the_shared_full_model(ablation_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import ablations as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    shared = train_shared_span(ablation_cfg, 0)
    frame = run_model_ablations(ablation_cfg, 0, shared=shared)
    assert list(frame["arm"]) == [f"model:{n}" for n, _ in MODEL_ABLATIONS]
    assert set(frame["kind"]) == {"training"}
    # `full` is the shared run, so it costs no extra training time.
    full = frame[frame["ablation"] == "full"].iloc[0]
    assert full["wall_seconds"] < float(frame["wall_seconds"].max())


def test_disabling_2d_pos_reduces_the_parameter_count(ablation_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import ablations as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    shared = train_shared_span(ablation_cfg, 0)
    frame = run_model_ablations(ablation_cfg, 0, shared=shared).set_index("ablation")
    assert frame.loc["no_2d_pos", "n_params"] < frame.loc["full", "n_params"]


def test_summarise_ablations_divides_by_the_noise_scale(tmp_path, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import ablations as module

    rows = []
    for seed in (0, 1, 2):
        rows.append({
            "arm": "verify:full", "seed": seed, "ablation": "full", "kind": "inference",
            "switch": "none", "strict_accuracy": 0.90 + 0.01 * seed, "coverage": 0.85,
            "canonical_accuracy": 0.95, "grounding_exact": 0.96, "hallucination_rate": 0.0,
        })
        rows.append({
            "arm": "verify:no_arithmetic", "seed": seed, "ablation": "no_arithmetic",
            "kind": "inference", "switch": "arithmetic=False",
            "strict_accuracy": 0.80 + 0.01 * seed, "coverage": 0.95,
            "canonical_accuracy": 0.85, "grounding_exact": 0.96, "hallucination_rate": 0.0,
        })
    pd.DataFrame(rows).to_csv(tmp_path / "ablation_runs.csv", index=False)
    monkeypatch.setattr(module, "TABLES", tmp_path)

    out = summarise_ablations(tables_dir=tmp_path)
    row = out[
        (out["ablation"] == "no_arithmetic") & (out["metric"] == "strict_accuracy")
    ].iloc[0]
    assert row["delta"] == pytest.approx(-0.10)
    assert row["noise_scale"] == pytest.approx(math.sqrt(2.0) * np.std([0.90, 0.91, 0.92], ddof=1))
    assert row["verdict"] in {"robust", "survives", "suggestive", "inside noise"}
    assert (tmp_path / "ablations.csv").exists()


def test_summarise_ablations_skips_a_family_with_no_full_row(tmp_path, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import ablations as module

    pd.DataFrame([{
        "arm": "verify:x", "seed": 0, "ablation": "x", "kind": "inference",
        "switch": "s", "strict_accuracy": 0.5,
    }]).to_csv(tmp_path / "ablation_runs.csv", index=False)
    monkeypatch.setattr(module, "TABLES", tmp_path)
    assert summarise_ablations(tables_dir=tmp_path).empty


# --- efficiency ------------------------------------------------------------

@pytest.fixture
def bench_prepared(tiny_cfg):  # noqa: ANN001, ANN201
    from gdx.pipelines.core import prepare

    return prepare(tiny_cfg, seed=0)


def test_head_cost_columns_and_ratios(tiny_cfg, bench_prepared, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import efficiency as module

    monkeypatch.setattr(module, "N_WARMUP", 2)
    monkeypatch.setattr(module, "N_REPEATS", 5)
    frame = head_cost(tiny_cfg, bench_prepared)
    assert set(frame["arm"]) == {"span_verify", "generative"}
    assert set(frame["batch_size"]) == {1, 8}
    for col in ("params", "mmacs", "latency_ms", "macs_per_ms",
                "latency_reduction_vs_generative"):
        assert col in frame.columns
        assert frame[col].notna().all()
    # The generative rows are the reference, so their ratio is exactly 1.
    gen = frame[frame["head"] == "generative"]
    assert np.allclose(gen["latency_reduction_vs_generative"], 1.0)


def test_head_cost_span_head_is_smaller(tiny_cfg, bench_prepared, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import efficiency as module

    monkeypatch.setattr(module, "N_WARMUP", 2)
    monkeypatch.setattr(module, "N_REPEATS", 5)
    frame = head_cost(tiny_cfg, bench_prepared).set_index(["head", "batch_size"])
    assert frame.loc[("span", 1), "params"] < frame.loc[("generative", 1), "params"]
    assert frame.loc[("span", 1), "head_params"] < frame.loc[("generative", 1), "head_params"]


def test_head_cost_encoder_macs_are_identical_across_heads(tiny_cfg, bench_prepared, monkeypatch):  # noqa: ANN001
    """The encoder is held fixed; only the head differs."""
    from gdx.pipelines import efficiency as module

    monkeypatch.setattr(module, "N_WARMUP", 2)
    monkeypatch.setattr(module, "N_REPEATS", 5)
    frame = head_cost(tiny_cfg, bench_prepared)
    for bs in (1, 8):
        rows = frame[frame["batch_size"] == bs]
        assert rows["encoder_mmacs"].nunique() == 1


def test_decode_scaling_fits_an_exponent(tiny_cfg, bench_prepared, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import efficiency as module

    monkeypatch.setattr(module, "N_WARMUP", 2)
    monkeypatch.setattr(module, "N_REPEATS", 4)
    frame = decode_scaling(tiny_cfg, bench_prepared, lengths=(4, 8, 16))
    assert set(frame["head"]) == {"span", "generative"}
    assert frame["fitted_exponent"].notna().all()
    # One exponent per head, repeated across that head's rows.
    for _head, group in frame.groupby("head"):
        assert group["fitted_exponent"].nunique() == 1


def test_decode_scaling_span_is_flatter_than_generative(tiny_cfg, bench_prepared, monkeypatch):  # noqa: ANN001
    """Selection's decode does not depend on value length; generation's does."""
    from gdx.pipelines import efficiency as module

    monkeypatch.setattr(module, "N_WARMUP", 3)
    monkeypatch.setattr(module, "N_REPEATS", 6)
    frame = decode_scaling(tiny_cfg, bench_prepared, lengths=(4, 12, 24))
    span = float(frame[frame["head"] == "span"]["fitted_exponent"].iloc[0])
    gen = float(frame[frame["head"] == "generative"]["fitted_exponent"].iloc[0])
    assert span < gen


def test_baseline_cost_reports_latency_and_tokens(tiny_cfg, bench_prepared):  # noqa: ANN001
    frame = baseline_cost(tiny_cfg, bench_prepared, n_docs=4)
    assert set(frame["arm"]) == {"heuristic", "llm_stub", "span_verify"}
    heuristic = frame[frame["arm"] == "heuristic"].iloc[0]
    assert heuristic["metric"] == "latency_per_doc_ms"
    assert heuristic["value"] > 0
    llm = frame[frame["arm"] == "llm_stub"].iloc[0]
    assert llm["metric"] == "prompt_tokens_per_doc"
    assert llm["value"] > 100
    local = frame[frame["arm"] == "span_verify"].iloc[0]
    assert local["value"] == 0.0


def test_baseline_cost_notes_state_the_measurement_conditions(tiny_cfg, bench_prepared):  # noqa: ANN001
    frame = baseline_cost(tiny_cfg, bench_prepared, n_docs=4)
    assert frame["note"].str.len().min() > 10
    assert any("lower bound" in n for n in frame["note"])


def test_torch_thread_note_records_the_protocol():
    note = torch_thread_note()
    assert note["torch_threads"] <= 4
    assert note["n_warmup"] >= 8
    assert note["n_repeats"] >= 25


# --- determinism pipeline --------------------------------------------------

def test_determinism_check_reports_exact_agreement(tiny_cfg, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx.pipelines import determinism as module

    monkeypatch.setattr(module, "TABLES", tmp_path)
    frame = module.check_determinism(tiny_cfg, seed=0)
    assert set(frame["quantity"]) == {
        "generator_token_boxes", "generator_token_texts",
        "span_verify_confidence", "span_verify_emitted_values",
    }
    assert frame["identical"].all()
    finite = frame["max_abs_difference"].dropna()
    assert (finite == 0.0).all()
    assert (tmp_path / "determinism.csv").exists()
