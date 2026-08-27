# Reproducibility

Everything in `results/` was produced by the commands below, on the machine
described below, with **no network access and no API keys**. The default path
downloads nothing.

## Environment

| | |
|---|---|
| OS | Windows 10 Pro 19045 |
| CPU | 4 cores, 16 GB RAM, no GPU |
| Python | 3.12.13 (pinned; **not** 3.13/3.14) |
| torch | 2.13.0+cpu |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| pandas | 3.0.5 |
| threads | `torch.set_num_threads(2)`, `OMP_NUM_THREADS=2` |

> **Python 3.12, not 3.13/3.14.** The 3.14 torch wheels in this environment fail
> to import on a missing bundled `torchgen`, and the PyPI `torchgen` package is an
> unrelated stub that does not fix it.

The thread cap is not cosmetic. Every latency number in `results/tables/` was
measured at two threads, and `scripts/_bootstrap.py` sets the environment
variables *before* importing torch, because BLAS backends read them at their own
first use and setting them afterwards is silently ignored.

### Setup

```bash
git clone https://github.com/arslan-ahmad/grounded-document-extraction
cd grounded-document-extraction

uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python ./.venv/Scripts/python.exe \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple torch
uv pip install --python ./.venv/Scripts/python.exe \
  numpy scipy pandas pillow pyyaml matplotlib pytest ruff \
  nbformat nbconvert ipykernel
uv pip install --python ./.venv/Scripts/python.exe -e . --no-deps
```

On Linux/macOS the interpreter is `./.venv/bin/python` and `make setup`,
`make smoke`, `make bench`, `make all`, `make test` do the same things.

## Reproducing the results

```bash
# 0. Prove the pipeline works end to end (~1 minute, real training, real metrics)
python scripts/train.py --config configs/smoke.yaml

# 1. Cost: params, MACs, measured latency, decode scaling, LLM token cost
python scripts/benchmark_efficiency.py

# 2. The seven-arm comparison, one invocation per seed
python scripts/run_experiments.py --seed 0
python scripts/run_experiments.py --seed 1
python scripts/run_experiments.py --seed 2

# 3. Ablations: inference-time switches share one checkpoint per seed
python scripts/run_ablations.py --seed 0
python scripts/run_ablations.py --seed 1
python scripts/run_ablations.py --seed 2 --summarise

# 4. Every derived table, from the committed per-item CSVs (runs no model)
python scripts/analyse.py --seeds 0 1 2

# 5. Figures, then inject the tables into the documentation
python scripts/make_figures.py
python scripts/render_docs.py

# 6. Notebooks
python scripts/build_notebooks.py --execute

# 7. Tests
python -m pytest tests -q -m "not slow"
python -m ruff check src scripts tests
```

`scripts/run_all.py` does stages 1-5 in order and **skips any stage whose output
already landed**, so an interrupted run is resumed rather than restarted. Pass
`--force` to redo one.

## Determinism

Two separate invocations, per-item outputs diffed:

<!-- table:determinism -->
<!-- /table -->

The generator is a pure function of `(doc_id, seed, config)`, and the evaluation
of a fixed checkpoint is deterministic. String-valued quantities report `0.0` when
identical and `NaN` otherwise, because there is no metric on strings and a
fabricated one would be worse than an honest gap.

Two design choices make this hold:

* **`zlib.crc32`, not `hash`.** Python's `hash` on `str` is salted per process, so
  a hash-bucketed vocabulary built with it would produce different features in
  different processes. Nothing about that failure announces itself.
* **Splits are disjoint id ranges, not a shuffled pool.** A document is
  regenerated rather than stored, so nothing can leak between splits because
  nothing is shared between them.

## Compute, and what it costs the results

Wall-clock measured on this machine at two threads.

| stage | wall-clock |
|---|---|
| `configs/smoke.yaml` end to end | ~35 s |
| span head, 1200 documents, 6 epochs | ~3.5 min |
| generative head, same budget | ~8 min |
| one seed of the seven-arm matrix (both heads + all evaluations) | ~15 min |
| one seed of the ablation matrix (3 trainings + 9 evaluations) | ~14 min |
| efficiency benchmark (latency + decode scaling + token cost) | ~2.5 min |
| `scripts/analyse.py` | ~10 s |
| notebooks 01-04, executed | ~9 min |
| **total for everything in `results/`** | **~95 min** |

The scale is set by that budget, and the honest consequence is stated here: this
is a **small** study. 1200 training documents, a 64-wide 2-layer encoder, three
seeds. The structural claims (P1, the arithmetic check, determinism) are
scale-free and hold as stated. The *accuracy* comparisons are the ones a larger
study could move, and `docs/RESULTS.md` places each one against the measured
run-to-run noise rather than asserting it.

Notebook 05 configures the same code at roughly 8x the scale for a GPU runtime and
is **not executed here**; it ships without outputs for that reason.

## Measurement pitfalls found while building this

Recorded because each one produced a plausible wrong number before it was fixed.

**Under-warmed latency.** With two warm-up iterations the span head measured far
slower than its steady state. The shipped benchmark uses 10 warm-up and 30 timed
repeats and reports median with IQR;
`test_committed_efficiency_used_enough_warmup` enforces the floor.

**A pure-Python decode loop.** The first span decoder looped over the
`(start, end)` grid in Python and was ~30x slower than the vectorised version.
That is a *measurement* bug, not just a slow one: it would have made the
selection head look slower than the generative head and inverted the efficiency
claim.

**A permissive amount parser.** An early `normalise_amount` stripped every
non-digit character, so `"Net Amount EUR 30,614.90"` parsed as `30614.90`. That
turned label text into a valid amount and inflated the count of document spans
that "contain" a value — which weakens the hallucination metric *in this method's
favour*. The parser now rejects residual letters and a second numeric group.

**A permissive date parser.** `normalise_date` ignored unrecognised words, so
`"Issued 29th of March 2024"` parsed to the same ISO date as
`"29th of March 2024"` and the label token counted as part of a correct grounding.
It now rejects any word that is not a month, a day, a year or a connective.

**Page-crossing spans.** See `docs/METHOD.md` §2.1 and
`docs/RESULTS.md` §9.1. The invariant test caught a real violation of the
repository's central guarantee on 3 of 8 seeds.

**A label matcher that stopped at the first mismatch.** The rule baseline's
synonym loop broke out after the first non-matching synonym, so only the longest
synonym was ever tried and the baseline found labels on 20% of fields instead of
96%. Reporting that as the rule baseline's ceiling would have been a strawman.

## Provenance of every number

* `results/tables/*.csv` — produced by the commands above; nothing is hand-edited.
* `results/runs/seed<k>/` — the resolved `config.yaml`, per-epoch
  `history.jsonl` for each head, `per_item.csv`, `summary.csv`, `summary.json`.
* `results/figures/*.png` — drawn from those CSVs by `scripts/make_figures.py`.
* Every table in `README.md` and `docs/*.md` is injected by
  `scripts/render_docs.py` from a CSV. **No number in the documentation is typed
  by hand**, and `tests/test_report_and_cli.py::test_shipped_documents_are_not_stale`
  fails if any of them has drifted.

Checkpoints (`*.pt`) are excluded from git: they are large and derived. Everything
needed to recompute the statistics — the per-item CSVs — is committed.
