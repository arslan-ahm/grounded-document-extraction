"""The documentation-injection mechanism, the CLI, the figures, the real loader.

If the injection mechanism breaks silently, the numbers in ``README.md`` stop
matching the CSVs and the repository's central promise -- every number traceable
to a committed artefact -- is void. So the marker parser, the staleness check, and
**the real documents** are all tested: ``test_shipped_documents_are_not_stale``
runs the same check CI runs, over every table in every shipped Markdown file.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):  # noqa: ANN202
    # The scripts import `_bootstrap` from their own directory, so that directory
    # has to be importable before the module body runs.
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def render_docs():  # noqa: ANN201
    return _load_script("render_docs")


# --- report rendering ------------------------------------------------------

def test_to_markdown_of_an_empty_frame_says_not_measured():
    from gdx.report import to_markdown

    assert to_markdown(pd.DataFrame()) == "_not measured_"
    assert to_markdown(None) == "_not measured_"


def test_to_markdown_of_unknown_columns_says_not_measured():
    from gdx.report import to_markdown

    assert to_markdown(pd.DataFrame([{"a": 1}]), ["b"]) == "_not measured_"


def test_to_markdown_shape_and_header():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"a": 1, "b": 2.5}, {"a": 3, "b": 4.0}]))
    lines = text.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert len(lines) == 4


def test_to_markdown_renders_nan_as_not_available():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"a": float("nan")}]))
    assert "n/a" in text


def test_to_markdown_bolds_the_maximum():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"a": 1.0}, {"a": 2.0}]), bold_max="a")
    assert "**2**" in text


def test_to_markdown_bolds_the_minimum():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"a": 1.0}, {"a": 2.0}]), bold_min="a")
    assert "**1**" in text


def test_to_markdown_renames_headers():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"a": 1}]), rename={"a": "Alpha"})
    assert text.splitlines()[0] == "| Alpha |"


def test_to_markdown_formats_booleans_as_words():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"ok": True}, {"ok": False}]))
    assert "yes" in text and "no" in text


def test_to_markdown_uses_scientific_notation_for_extremes():
    from gdx.report import to_markdown

    text = to_markdown(pd.DataFrame([{"v": 1.5e-7}, {"v": 4.2e6}]))
    assert "e-07" in text or "e-7" in text
    assert "e+06" in text or "e+6" in text


def test_every_registered_table_renders_without_a_csv(tmp_path, monkeypatch):  # noqa: ANN001
    """A missing CSV must produce ``not measured``, never an invented row."""
    from gdx import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    for name, renderer in report.ALL.items():
        text = renderer()
        assert text == "_not measured_", f"{name} invented content without a CSV"


def test_registered_table_names_are_lowercase_with_underscores():
    """The marker regex only matches ``[a-z_]+``."""
    from gdx.report import ALL

    for name in ALL:
        assert name.replace("_", "").isalpha()
        assert name.islower()


# --- render_docs -----------------------------------------------------------

def test_marker_pattern_matches_a_block(render_docs):  # noqa: ANN001
    text = "before\n<!-- table:method -->\nold\n<!-- /table -->\nafter"
    assert render_docs.PATTERN.search(text) is not None


def test_render_replaces_the_body(render_docs, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    out, changed = render_docs.render("<!-- table:method -->\nSTALE\n<!-- /table -->")
    assert "STALE" not in out
    assert "_not measured_" in out
    assert changed == ["method"]


def test_render_is_idempotent(render_docs, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = "<!-- table:method -->\n_not measured_\n<!-- /table -->"
    out, changed = render_docs.render(text)
    assert changed == []
    assert out == text


def test_render_rejects_an_unknown_marker(render_docs):  # noqa: ANN001
    with pytest.raises(KeyError, match="unknown table"):
        render_docs.render("<!-- table:not_a_table -->\n\n<!-- /table -->")


def test_render_handles_several_blocks(render_docs, tmp_path, monkeypatch):  # noqa: ANN001
    from gdx import report

    monkeypatch.setattr(report, "TABLES", tmp_path)
    text = (
        "<!-- table:method -->\nx\n<!-- /table -->\n"
        "middle\n"
        "<!-- table:efficiency -->\ny\n<!-- /table -->"
    )
    out, changed = render_docs.render(text)
    assert set(changed) == {"method", "efficiency"}
    assert out.count("_not measured_") == 2


def test_shipped_documents_are_not_stale(render_docs):
    """The check CI runs: every table in every shipped document is current.

    This is what makes the documentation trustworthy -- a number that drifted from
    its CSV is a test failure rather than something a reader has to catch.
    """
    stale = {}
    for rel in render_docs.TARGETS:
        path = ROOT / rel
        if not path.exists():
            continue
        _, changed = render_docs.render(path.read_text(encoding="utf-8"))
        if changed:
            stale[rel] = changed
    assert not stale, f"stale tables: {stale}; run scripts/render_docs.py"


def test_every_shipped_document_uses_at_least_one_table_marker(render_docs):
    """A document with no markers would be quoting hand-typed numbers."""
    found = 0
    for rel in render_docs.TARGETS:
        path = ROOT / rel
        if path.exists():
            found += len(render_docs.PATTERN.findall(path.read_text(encoding="utf-8")))
    assert found > 8, f"only {found} injected tables across the documentation"


# --- committed artefacts ---------------------------------------------------

def _tables() -> Path:
    return ROOT / "results" / "tables"


@pytest.mark.parametrize(
    "name",
    ["method_comparison", "seed_variance", "verdicts", "statistical_tests",
     "normalisation_cost", "calibration", "abstention", "efficiency",
     "decode_scaling", "baseline_cost", "per_field", "ablations", "determinism"],
)
def test_committed_table_exists_and_is_populated(name):  # noqa: ANN001
    path = _tables() / f"{name}.csv"
    assert path.exists(), f"{name}.csv is missing; run scripts/run_all.py"
    frame = pd.read_csv(path)
    assert not frame.empty, f"{name}.csv is empty"


def test_committed_seed_runs_covers_three_seeds():
    frame = pd.read_csv(_tables() / "seed_runs.csv")
    assert frame["seed"].nunique() >= 3
    assert frame["arm"].nunique() == 7


def test_committed_method_comparison_has_every_arm():
    frame = pd.read_csv(_tables() / "method_comparison.csv")
    assert set(frame["arm"]) == {
        "heuristic", "llm_stub", "generative", "generative_verify",
        "span_only", "span_verify", "span_verify_norm",
    }


def test_committed_span_arms_have_zero_hallucination_rate():
    """The headline claim, checked against the committed evidence."""
    frame = pd.read_csv(_tables() / "seed_runs.csv")
    span = frame[frame["arm"].isin(["span_only", "span_verify"])]
    assert not span.empty
    assert (span["hallucination_rate"].fillna(0.0) == 0.0).all()


def test_committed_determinism_is_exact():
    frame = pd.read_csv(_tables() / "determinism.csv")
    assert frame["identical"].all(), "a repeated run did not reproduce"
    finite = frame["max_abs_difference"].dropna()
    assert (finite == 0.0).all()


def test_committed_efficiency_used_enough_warmup():
    """Under-warmed timing is the measurement error the standard calls out."""
    frame = pd.read_csv(_tables() / "efficiency.csv")
    assert (frame["n_warmup"] >= 8).all()
    assert (frame["n_repeats"] >= 25).all()


def test_committed_figures_exist():
    figures = ROOT / "results" / "figures"
    names = {p.stem for p in figures.glob("*.png")}
    assert {"document", "hallucination_and_coverage", "decode_scaling"} <= names


def test_committed_run_directories_carry_their_config():
    for run_dir in sorted((ROOT / "results" / "runs").glob("seed*")):
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "summary.json").exists()
        payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert "config" in payload
        assert payload["splits"]["test"] > 0


# --- CLI -------------------------------------------------------------------

def test_cli_version_exits_zero(capsys):  # noqa: ANN001
    from gdx.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0


def test_cli_requires_a_command():
    from gdx.cli import main

    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_an_unknown_command():
    from gdx.cli import main

    with pytest.raises(SystemExit):
        main(["fly"])


def test_cli_data_command_prints_the_document(capsys):  # noqa: ANN001
    from gdx.cli import main

    assert main(["data", "--config", "configs/base.yaml", "--doc-id", "1"]) == 0
    out = capsys.readouterr().out
    assert "INVOICE" in out
    assert "invoice_id" in out
    assert "tokens" in out


def test_cli_data_respects_a_set_override(capsys):  # noqa: ANN001
    from gdx.cli import main

    main(["data", "--config", "configs/base.yaml", "--set", "data.multi_page_prob=1.0"])
    assert "page(s)" in capsys.readouterr().out


def test_cli_parser_lists_every_documented_command():
    from gdx.cli import build_parser

    actions = build_parser()._subparsers._group_actions[0].choices  # noqa: SLF001
    assert set(actions) == {
        "train", "experiments", "ablations", "benchmark", "analyse", "figures", "data"
    }


# --- figures ---------------------------------------------------------------

def test_figure_document_writes_a_png(tmp_path, doc):  # noqa: ANN001
    from gdx.viz import figure_document

    path = figure_document(doc, tmp_path)
    assert path is not None and path.exists()
    assert path.stat().st_size > 1000


def test_figures_return_none_without_their_csv(tmp_path, monkeypatch):  # noqa: ANN001
    """A missing table must skip the figure, not draw an empty one."""
    from gdx import viz

    monkeypatch.setattr(viz, "TABLES", tmp_path)
    assert viz.figure_hallucination(tmp_path) is None
    assert viz.figure_normalisation_cost(tmp_path) is None
    assert viz.figure_efficiency(tmp_path) is None
    assert viz.figure_seed_variance(tmp_path) is None


def test_no_backend_is_forced_at_import_time():
    """Forcing Agg at import makes every notebook plot render blank."""
    source = (ROOT / "src" / "gdx" / "viz.py").read_text(encoding="utf-8")
    header = source.split("def ")[0]
    assert "matplotlib.use" not in header
    assert "ipykernel" in source


# --- optional real-data loader --------------------------------------------

def test_real_dataset_root_rejects_an_unknown_name():
    from gdx.data.real import dataset_root

    with pytest.raises(ValueError, match="unsupported real dataset"):
        dataset_root("data/raw", "mnist")


def test_real_dataset_root_raises_when_absent(tmp_path):  # noqa: ANN001
    from gdx.data.real import RealDatasetUnavailable, dataset_root

    with pytest.raises(RealDatasetUnavailable, match="download_real"):
        dataset_root(tmp_path, "funsd")


def test_real_loader_reads_a_jsonl_dump(tmp_path):  # noqa: ANN001
    from gdx.data.real import load_jsonl_documents

    payload = {
        "id": 7,
        "width": 100.0,
        "height": 200.0,
        "tokens": [
            {"text": "Total", "box": [10, 20, 40, 30]},
            {"text": "120.00", "box": [50, 20, 90, 30]},
        ],
        "fields": {"total": "120.00"},
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    docs = load_jsonl_documents(path, "cord")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == 7
    assert doc.meta["approximate_provenance"] is True
    assert doc.fields["total"].present
    assert doc.fields["total"].span == (1, 1)
    assert 0.0 <= doc.tokens[0].box[0] <= 1.0


def test_real_loader_marks_an_unmatchable_value_absent(tmp_path):  # noqa: ANN001
    """An annotated value the tokens do not contain cannot be selected at all."""
    from gdx.data.real import load_jsonl_documents

    payload = {
        "id": 1,
        "width": 10.0,
        "height": 10.0,
        "tokens": [{"text": "Total", "box": [1, 1, 3, 2]}],
        "fields": {"total": "999.00"},
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    doc = load_jsonl_documents(path, "cord")[0]
    assert not doc.fields["total"].present


def test_real_loader_applies_the_dataset_alias(tmp_path):  # noqa: ANN001
    from gdx.data.real import load_jsonl_documents

    payload = {
        "id": 1,
        "width": 10.0,
        "height": 10.0,
        "tokens": [{"text": "5.00", "box": [1, 1, 3, 2]}],
        "fields": {"total.total_price": "5.00"},
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    doc = load_jsonl_documents(path, "cord")[0]
    assert doc.fields["total"].present


def test_real_loader_rejects_a_zero_page_size(tmp_path):  # noqa: ANN001
    from gdx.data.real import load_jsonl_documents

    payload = {
        "id": 1, "width": 0.0, "height": 10.0,
        "tokens": [{"text": "x", "box": [1, 1, 3, 2]}], "fields": {},
    }
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="page size"):
        load_jsonl_documents(path, "cord")


def test_real_loader_raises_on_a_missing_file(tmp_path):  # noqa: ANN001
    from gdx.data.real import RealDatasetUnavailable, load_jsonl_documents

    with pytest.raises(RealDatasetUnavailable):
        load_jsonl_documents(tmp_path / "nope.jsonl", "cord")


# --- utils -----------------------------------------------------------------

def test_latency_benchmark_rejects_bad_counts():
    from gdx.utils.latency import benchmark_callable

    with pytest.raises(ValueError, match="n_repeats"):
        benchmark_callable(lambda: None, 1, 0)
    with pytest.raises(ValueError, match="n_warmup"):
        benchmark_callable(lambda: None, -1, 5)


def test_latency_benchmark_reports_median_and_iqr():
    from gdx.utils.latency import benchmark_callable

    result = benchmark_callable(lambda: sum(range(100)), 8, 25)
    assert result.n_warmup == 8 and result.n_repeats == 25
    assert result.median_ms >= 0.0
    assert result.min_ms <= result.median_ms <= result.max_ms
    assert not math.isnan(result.iqr_ms)


def test_fit_power_exponent_recovers_a_known_exponent():
    from gdx.utils.latency import fit_power_exponent

    sizes = [1.0, 2.0, 4.0, 8.0]
    times = [s**2.0 for s in sizes]
    assert fit_power_exponent(sizes, times) == pytest.approx(2.0, abs=1e-9)


def test_fit_power_exponent_of_a_constant_is_zero():
    from gdx.utils.latency import fit_power_exponent

    assert fit_power_exponent([1.0, 2.0, 4.0], [5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_fit_power_exponent_needs_two_points():
    from gdx.utils.latency import fit_power_exponent

    assert math.isnan(fit_power_exponent([1.0], [1.0]))
    assert math.isnan(fit_power_exponent([], []))


def test_seed_everything_rejects_a_negative_seed():
    from gdx.utils.seed import seed_everything

    with pytest.raises(ValueError, match="non-negative"):
        seed_everything(-1)


def test_seed_everything_makes_torch_reproducible():
    import torch

    from gdx.utils.seed import seed_everything

    seed_everything(5)
    a = torch.rand(4)
    seed_everything(5)
    assert torch.equal(a, torch.rand(4))


def test_temporary_seed_restores_the_stream():
    import torch

    from gdx.utils.seed import seed_everything, temporary_seed

    seed_everything(1)
    torch.rand(1)
    state = torch.get_rng_state().clone()
    with temporary_seed(99):
        torch.rand(10)
    assert torch.equal(torch.get_rng_state(), state)


def test_worker_seed_is_distinct_and_deterministic():
    from gdx.utils.seed import worker_seed

    seeds = {worker_seed(0, w) for w in range(8)}
    assert len(seeds) == 8
    assert worker_seed(3, 2) == worker_seed(3, 2)


def test_jsonl_logger_writes_and_reads(tmp_path):  # noqa: ANN001
    from gdx.utils.logging import JsonlLogger

    logger = JsonlLogger(tmp_path / "h.jsonl")
    logger.log(epoch=0, loss=1.5)
    logger.log(epoch=1, loss=float("nan"))
    records = JsonlLogger.read(tmp_path / "h.jsonl")
    assert len(records) == 2
    assert records[1]["loss"] is None  # NaN is not valid JSON


def test_jsonl_logger_read_of_a_missing_file_is_empty(tmp_path):  # noqa: ANN001
    from gdx.utils.logging import JsonlLogger

    assert JsonlLogger.read(tmp_path / "nope.jsonl") == []


def test_jsonl_logger_skips_malformed_lines(tmp_path):  # noqa: ANN001
    from gdx.utils.logging import JsonlLogger

    path = tmp_path / "h.jsonl"
    path.write_text('{"a": 1}\nnot json\n\n{"a": 2}\n', encoding="utf-8")
    assert [r["a"] for r in JsonlLogger.read(path)] == [1, 2]


def test_write_json_uses_lf_endings(tmp_path):  # noqa: ANN001
    from gdx.utils.logging import write_json

    path = write_json(tmp_path / "x.json", {"a": [1, 2], "b": float("inf")})
    assert path.read_bytes().count(b"\r\n") == 0
    assert json.loads(path.read_text(encoding="utf-8"))["b"] is None
