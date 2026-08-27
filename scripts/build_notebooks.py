"""Generate the notebooks with ``nbformat``, optionally executing them.

    python scripts/build_notebooks.py            # write the .ipynb files
    python scripts/build_notebooks.py --execute  # write, then run with outputs

Generating rather than hand-writing means the notebooks cannot drift from the
package API, and every cell gets a stable id so a re-generation produces a clean
diff. Execution is a separate flag because the GPU notebook is not meant to run
here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import _bootstrap
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = _bootstrap.ROOT
NOTEBOOKS = ROOT / "notebooks"

HEADER = """\
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))
import torch; torch.set_num_threads(2)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
print("torch", torch.__version__)
"""


def _cells(spec: list[tuple[str, str]], prefix: str) -> list:
    out = []
    for i, (kind, source) in enumerate(spec):
        cell = new_markdown_cell(source) if kind == "md" else new_code_cell(source)
        cell["id"] = f"{prefix}-{i:02d}"
        out.append(cell)
    return out


def notebook_01() -> list[tuple[str, str]]:
    return [
        ("md", """# 01 - The data, and why provenance is measurable here

Every field in a generated document carries the **token index range it was
written at**, because the generator placed it there. That is the oracle the
grounding metric is scored against, and no annotated real dataset provides it --
which is why the shipped results are synthetic and why this notebook starts with
the data rather than with the model.

Three things to look for below: the ground-truth spans point at *regions*, not
strings; the distractor amounts sit next to the targets; and some fields are
genuinely absent, so abstaining is the correct answer."""),
        ("code", HEADER),
        ("code", """from gdx.config import DataConfig
from gdx.data.generator import generate_dataset, generate_document

cfg = DataConfig()
doc = generate_document(3, cfg, seed=0)
print(f"{len(doc)} tokens across {doc.n_pages} page(s)\\n")
print(" ".join(doc.texts))"""),
        ("code", """rows = []
for name, truth in doc.fields.items():
    rows.append({
        "field": name,
        "present": truth.present,
        "span": truth.span,
        "written": truth.span_text,
        "target": truth.value,
        "needs_normalisation": truth.requires_normalisation,
        "other_occurrences": len(truth.alt_spans),
    })
pd.DataFrame(rows)"""),
        ("md", """### The document, drawn

Grey boxes are context; coloured boxes are ground-truth provenance. Notice the
summary block: `Subtotal`, the distractor lines, and `Total Due` are neighbours,
which is what makes "read the bottom-right number" fail."""),
        ("code", """from gdx.viz import figure_document
figure_document(doc, out_dir=__import__("pathlib").Path("../results/figures"), name="document")
from IPython.display import Image
Image(filename="../results/figures/document.png")"""),
        ("md", """### The difficulty is measured, not asserted

Each knob is a config field. The table shows what the shipped defaults produce
over 400 documents."""),
        ("code", """docs = generate_dataset(400, cfg, seed=0)
from gdx.data.schema import FIELDS
stats = []
for name in FIELDS:
    truths = [d.fields[name] for d in docs]
    present = [t for t in truths if t.present]
    stats.append({
        "field": name,
        "present_rate": len(present) / len(truths),
        "needs_normalisation_rate": sum(t.requires_normalisation for t in present) / max(1, len(present)),
        "mean_occurrences": np.mean([len(t.all_spans) for t in present]) if present else np.nan,
        "mean_span_tokens": np.mean([t.span[1] - t.span[0] + 1 for t in present]) if present else np.nan,
    })
pd.DataFrame(stats)"""),
        ("code", """print("tokens per document:", int(np.min([len(d) for d in docs])), "-", int(np.max([len(d) for d in docs])))
print("pages:", pd.Series([d.n_pages for d in docs]).value_counts().to_dict())
print("distractor amounts per document:", float(np.mean([d.meta["n_distractors"] for d in docs])).__round__(2))
print("documents where the total is printed twice:", float(np.mean([d.meta["duplicate_total"] for d in docs])).__round__(3))"""),
        ("md", """### The invariant, on data alone

Every admissible span of every document is, by construction, present in that
document. This is the property the whole repository rests on, and it is a fact
about the *output space* rather than about any model."""),
        ("code", """from gdx.data.schema import MAX_VALUE_SPAN, canonical_value
checked = violations = 0
for d in docs[:20]:
    n = len(d.tokens)
    for s in range(n):
        page = d.tokens[s].page
        for e in range(s, min(s + MAX_VALUE_SPAN, n)):
            if d.tokens[e].page != page:
                break
            text = d.span_text(s, e)
            if canonical_value("total", text):
                checked += 1
                violations += not d.contains_value("total", text)
print(f"{checked} spans checked, {violations} not found in their own document")"""),
        ("md", """**The exception that proves the rule.** A span whose endpoints straddle a page
break is a sequence of document tokens but *not* a contiguous region: it has no
box, and the provenance predicate refuses it. That is why span decoding confines
candidates to one page -- and the invariant test caught the bug when it did
not."""),
        ("code", """multi = next(d for d in docs if d.n_pages > 1)
boundary = next(i for i in range(1, len(multi.tokens)) if multi.tokens[i].page != multi.tokens[i - 1].page)
spliced = multi.span_text(boundary - 1, boundary)
print("spliced across the page break:", repr(spliced))
print("union box:", multi.union_box(boundary - 1, boundary))
print("found by the provenance check:", multi.contains_value("vendor_name", spliced))"""),
    ]


def notebook_02() -> list[tuple[str, str]]:
    return [
        ("md", """# 02 - Selection versus generation, on one encoder

The comparison that isolates the claim: one encoder, one training loop, one seed,
one dataset, and **only the output interface differs**. `span` emits two indices;
`generative` emits characters over a closed 70-character alphabet.

This notebook trains both at a reduced scale so it runs in a few minutes. The
shipped numbers come from `scripts/run_experiments.py` at full scale; the tables
read from `results/tables/` at the bottom are those."""),
        ("code", HEADER),
        ("code", """from gdx.config import load_config
from gdx.pipelines.core import prepare, train_head
from gdx.arms import ARM_BY_NAME, model_candidates
from gdx.pipelines.core import evaluate_arm

cfg = load_config("../configs/base.yaml", ["data.n_train=400", "data.n_val=100", "data.n_test=150", "optim.epochs=4"])
prepared = prepare(cfg, seed=0)
print(prepared.splits.sizes)"""),
        ("code", """models, candidates = {}, {}
for head in ("span", "generative"):
    models[head], hist = train_head(cfg, prepared, head)
    candidates[head] = model_candidates(models[head], prepared.splits.test, cfg)
    print(head, "params", hist["n_params"], "train seconds", round(hist["train_seconds"], 1))"""),
        ("code", """rows = []
for arm_name in ("generative", "generative_verify", "span_only", "span_verify", "span_verify_norm"):
    arm = ARM_BY_NAME[arm_name]
    result = evaluate_arm(arm, prepared.splits.test, candidates[arm.source], cfg)
    rows.append({
        "arm": arm_name,
        "strict": result.summary["strict_accuracy"],
        "canonical": result.summary["canonical_accuracy"],
        "coverage": result.summary["coverage"],
        "hallucination": result.summary["hallucination_rate"],
        "grounding_exact": result.summary["grounding_exact"],
    })
pd.DataFrame(rows).round(4)"""),
        ("md", """The `hallucination` column is the point. For the span arms it is exactly 0 and
cannot be otherwise. For `generative` it is whatever the decoder produced.

`generative_verify` is the arm a fair reading demands: the *same* verification
loop applied to generated strings. If it also reaches 0, then the check rather
than the head is what removes hallucinated values -- and that is reported rather
than avoided."""),
        ("code", """# What the generative head actually emits, next to the truth.
gen = models["generative"]
batch = prepared.splits.test.collate(list(range(6)))
preds = gen.predict(batch)
rows = []
for doc, per_doc in zip(batch.docs, preds):
    for pred in per_doc:
        truth = doc.fields[pred.field_name]
        rows.append({
            "doc": doc.doc_id, "field": pred.field_name,
            "generated": pred.text, "target": truth.value,
            "in_document": doc.contains_value(pred.field_name, pred.text) if pred.text else None,
        })
pd.DataFrame(rows).head(24)"""),
        ("md", """### The shipped tables

Everything below is read from `results/tables/`, produced by
`scripts/run_experiments.py` and `scripts/analyse.py` at full scale over three
seeds."""),
        ("code", """import pathlib
T = pathlib.Path("../results/tables")
pd.read_csv(T / "method_comparison.csv")"""),
        ("code", """pd.read_csv(T / "verdicts.csv")"""),
        ("code", """pd.read_csv(T / "statistical_tests.csv")"""),
    ]


def notebook_03() -> list[tuple[str, str]]:
    return [
        ("md", """# 03 - Ablations and the noise floor

No ablation delta means anything until it is placed against the run-to-run scale
of a difference, `sqrt(2) * sd` from the seed study. This notebook reads the
committed seed study, computes that scale, and then reads the ablation table --
which has already had the same division applied."""),
        ("code", HEADER),
        ("code", """import pathlib
T = pathlib.Path("../results/tables")
runs = pd.read_csv(T / "seed_runs.csv")
runs[["arm", "seed", "strict_accuracy", "canonical_accuracy", "coverage", "hallucination_rate"]]"""),
        ("code", """pd.read_csv(T / "seed_variance.csv")"""),
        ("md", """### Inference-time ablations reuse one checkpoint

Verification on/off, type check on/off, arithmetic on/off and abstention on/off
are all evaluated against the *same trained weights*, so their deltas contain no
initialisation noise at all. The training-time ablations (2-D position, spatial
bias) require a retrain and therefore do."""),
        ("code", """pd.read_csv(T / "ablations.csv")"""),
        ("code", """abl = pd.read_csv(T / "ablation_runs.csv")
abl[["arm", "seed", "kind", "switch", "strict_accuracy", "coverage", "hallucination_rate", "verify_arith_ok_frac"]]"""),
        ("md", """### The cost of the ideology, measured

Span selection cannot emit a value that requires normalisation. On a date written
"3rd of Jan 2024" against a target of `2024-01-03`, no contiguous span equals the
target, so strict accuracy there is bounded at zero. The table splits accuracy by
exactly that condition and reports how many fields it applies to."""),
        ("code", """pd.read_csv(T / "normalisation_cost.csv")"""),
        ("code", """from IPython.display import Image
Image(filename="../results/figures/normalisation_cost.png")"""),
    ]


def notebook_04() -> list[tuple[str, str]]:
    return [
        ("md", """# 04 - Abstention, calibration, and the verification loop

The interesting question is not how often each arm is right but whether it knows
when it is wrong -- and whether, when it does not know, it declines. On fields the
generator genuinely omitted, abstaining is the *correct* output."""),
        ("code", HEADER),
        ("code", """import pathlib
T = pathlib.Path("../results/tables")
pd.read_csv(T / "abstention.csv")"""),
        ("code", """pd.read_csv(T / "calibration.csv")"""),
        ("md", """`ece` uses equal-width bins and `ace` equal-mass ones. With confidences piled
near 1.0 most equal-width bins are nearly empty, so `ace` is the more trustworthy
of the two and both are shown so the disagreement is visible."""),
        ("code", """from IPython.display import Image
Image(filename="../results/figures/risk_coverage.png")"""),
        ("md", """### What the loop actually did

Every rejection is counted. `verify_arith_ok_frac` is the fraction of documents
where the arithmetic check *could* run and passed after repair; the check does not
apply at all when the generator dropped the subtotal or the tax, and that case is
`NaN` rather than a pass."""),
        ("code", """runs = pd.read_csv(T / "seed_runs.csv")
cols = ["arm", "seed", "verify_iters_per_doc", "verify_reselections_per_doc",
        "verify_arith_checked_frac", "verify_arith_ok_frac", "coverage"]
runs[[c for c in cols if c in runs.columns]]"""),
        ("md", """### Efficiency, measured

The axis is decode cost, and it is structural: a span head emits two indices in a
fixed number of sequential steps; a character decoder emits `L` characters in `L`
sequential steps per field. The exponent is fitted from measurements."""),
        ("code", """pd.read_csv(T / "efficiency.csv")"""),
        ("code", """pd.read_csv(T / "decode_scaling.csv")"""),
        ("code", """Image(filename="../results/figures/decode_scaling.png")"""),
        ("code", """pd.read_csv(T / "baseline_cost.csv")"""),
    ]


def notebook_05() -> list[tuple[str, str]]:
    return [
        ("md", """# 05 - Colab / Kaggle: the same code at full scale

Runs the identical package on a GPU runtime at roughly 8x the shipped scale:
6000 training documents, 288 tokens per document, a wider encoder, and a longer
schedule. Nothing here is a different implementation -- only the config changes.

There is also an optional real-dataset cell. Read the caveat: FUNSD, CORD and
SROIE annotate *values*, not the token span each value was read from, so on real
data the grounding oracle can only be approximated by string matching. No number
in `docs/RESULTS.md` comes from that path."""),
        ("md", "## Install"),
        ("code", """# In Colab or Kaggle:
# !git clone https://github.com/arslan-ahmad/grounded-document-extraction
# %cd grounded-document-extraction
# !pip install -q torch numpy scipy pandas pyyaml matplotlib
import sys, os
sys.path.insert(0, os.path.abspath("src") if os.path.isdir("src") else os.path.abspath("../src"))
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())"""),
        ("md", "## Full-scale configuration"),
        ("code", """from gdx.config import load_config
overrides = [
    "data.n_train=6000", "data.n_val=800", "data.n_test=1500",
    "data.max_tokens=288", "data.max_pages=3",
    "model.d_model=160", "model.n_layers=4", "model.n_heads=8", "model.d_ff=320",
    "model.dec_hidden=192",
    "optim.epochs=20", "optim.batch_size=32", "optim.lr=0.002",
    "run.n_threads=8",
]
cfg = load_config("configs/base.yaml", overrides)
print(cfg.to_dict()["model"])"""),
        ("md", "## Train both heads and evaluate every arm"),
        ("code", """from gdx.pipelines.experiments import run_seed
frame = run_seed(cfg, seed=0)
frame[["arm", "strict_accuracy", "canonical_accuracy", "coverage",
       "hallucination_rate", "grounding_exact"]]"""),
        ("md", """## The invariant at scale

The guarantee is structural, so scale cannot break it. This asserts rather than
inspects."""),
        ("code", """import pandas as pd
per_item = pd.read_csv("results/runs/seed0/per_item.csv")
span = per_item[per_item["arm"].str.startswith("span") & per_item["emitted"]]
print(f"{len(span)} emitted span values; ungrounded: {int((~span['grounded']).sum())}")
assert bool(span["grounded"].all())"""),
        ("md", "## Optional: a real dataset, with the caveat"),
        ("code", """# !python scripts/download_real.py --dataset cord   # prints the source and licence
# from gdx.data.real import load_real_dataset
# docs = load_real_dataset("cord", root="data/raw", split="test")
# print(len(docs), "documents; provenance recovered by string matching (approximate)")"""),
        ("md", "## Ablations and analysis at scale"),
        ("code", """from gdx.pipelines.ablations import run_model_ablations, run_verify_ablations, train_shared_span
shared = train_shared_span(cfg, 0)
run_verify_ablations(cfg, 0, shared=shared)
run_model_ablations(cfg, 0, shared=shared)
from gdx.pipelines.ablations import summarise_ablations
summarise_ablations()"""),
    ]


SPECS = {
    "01_data_and_provenance": notebook_01,
    "02_selection_vs_generation": notebook_02,
    "03_ablations_and_noise": notebook_03,
    "04_abstention_and_calibration": notebook_04,
    "05_colab_full_scale": notebook_05,
}

#: The GPU notebook is not executed here: it configures an 8x-scale run.
NO_EXECUTE = {"05_colab_full_scale"}


def build(name: str) -> Path:
    notebook = new_notebook(cells=_cells(SPECS[name](), name[:2]))
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / f"{name}.ipynb"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        nbformat.write(notebook, handle)
    return path


def execute(path: Path, timeout: int = 1800) -> int:
    cmd = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
        "--inplace", f"--ExecutePreprocessor.timeout={timeout}", str(path),
    ]
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["OMP_NUM_THREADS"] = "2"
    return subprocess.run(cmd, cwd=path.parent, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", default=None, help="build one notebook by name")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    names = [args.only] if args.only else list(SPECS)
    failures = 0
    for name in names:
        path = build(name)
        print(f"wrote {path}")
        if args.execute and name not in NO_EXECUTE:
            code = execute(path, args.timeout)
            print(f"  executed: exit {code}")
            failures += code != 0
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
