![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13%20cpu-red)
![Tests](https://img.shields.io/badge/tests-643%20passing-brightgreen)
![Params](https://img.shields.io/badge/params-0.109M-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# Where Did That Number Come From? — Grounded Document Extraction

**Field extraction from visually-rich documents as *selection plus verification*
rather than generation: every emitted value carries the token span it was read
from, and a value whose provenance does not verify is abstained on.**

**Efficiency axis — decode cost, measured.** Against the same encoder with a
generative head: **9.306322x lower end-to-end latency** (17.72595 ms against
164.9634 ms, batch 1, 154 tokens, 10 warm-up / 30 timed repeats), 7.258041x fewer
MACs, and a head of 1040 parameters against 93290. The gap is *asymptotic in value
length*: fitted decode-cost exponent **0.077604 for selection against 0.755079 for
generation**.

The approach this replaces runs a model over the page, has it emit a **string**
per field, and trusts the string. A hallucinated total and a correctly-read total
are then the same kind of object — both elements of `Σ*` — and there is no query
you can put to the output that tells them apart. This repository makes the output
a **pair of token indices** instead. The value is whatever those indices span, so
a string that appears nowhere in the document is not a low-probability output: it
is not in the output space at all.

> **Result, up front — including the parts that are not flattering.**
>
> **The guarantee holds, exactly, over 1075037 measured opportunities.**
> 1064592 enumerated admissible spans, 1598 values emitted by *untrained* span
> heads across 8 seeds, and 8847 emitted by the trained checkpoints across 3
> seeds — **0 ungrounded, in all three populations, with the verification loop
> switched off.** The generative reference approach on the same encoder, same
> data, same budget and same seed hallucinates at 0.998597. The control fires as
> it must: untrained generative heads score 1.000000.
>
> **But the checker, not the head, is what removes hallucinated values.** The same
> provenance check applied to the generative arm (`generative_verify`) also
> reaches 0.0 — by abstaining almost completely, at 0.001250 coverage against
> 0.890938. Selection's real contribution is that it satisfies the check *for
> free*, at 0.877188 coverage, and needs no checker to be safe. This is the single
> most important qualification in the repository, and it exists because the arm
> capable of refuting the broad claim was built and reported.
>
> **Selection beats the strong rule baseline.** +0.130208 canonical accuracy at
> **9.362311x** the run-to-run noise scale, +0.119583 coverage at 6.130715x. The
> rule baseline is not a strawman: all label synonyms, same-row *and* below-label
> geometry, best of six configurations chosen on validation — and it inherits the
> same grounding guarantee, because it also selects.
>
> **The verification loop buys grounding, not accuracy.** Against the identical
> weights with the loop off: exact-span grounding +0.026098 at 4.207149x
> (**robust**), coverage −0.044375 at 2.274987x (**a real cost**), and canonical
> accuracy +0.010208 at 0.734005x — **inside noise**. The loop does not
> demonstrably improve accuracy at this scale.
>
> **The ideology's cost is exactly zero where it applies.** On the 211 test fields
> written in a form that is not the target, *every* selection-based arm — this
> method, its verification-free ablation, and the rule baseline — scores
> **0.000000** strict accuracy, against 0.941682 on verbatim fields. A
> deterministic date parser recovers it to 0.995261 and **weakens the guarantee**,
> which is stated wherever that arm appears.
>
> **And the reference approach is undertrained at this budget.** `generative`
> reaches 0.047708 canonical accuracy with its validation loss plateauing near
> 1.001558 nats per character. The hallucination comparison is structural and does
> not depend on that; the *accuracy* comparison does, and §9.3 of
> [docs/RESULTS.md](docs/RESULTS.md) says so.
>
> Full tables, the seed study, every ablation and every retraction:
> **[docs/RESULTS.md](docs/RESULTS.md)**.

---

## The one-paragraph argument

An accounts-payable system is told the invoice total is `$36,737.88`. Against
what? The model emitted a string. It might have read the total, or the subtotal,
or the "Balance Forward" line two rows up, or nothing at all — and the output is
identical in all four cases. Ask instead for *indices*: `(118, 119)`. Now the
value has a location, so you can draw a box around it, you can check that
`subtotal + tax = total` across three separately-located numbers, and you can
notice when a field the document does not contain has been confidently supplied.
The cost is that selection cannot *transform* — it cannot turn "3rd of Jan 2024"
into `2024-01-03` — and this repository measures that cost instead of arranging
an evaluation where it never arises.

## What makes the grounding claim checkable

```
generator places every token         →  the token index range each value
                                        was written at                  exact
union of that range's boxes          →  the region it occupies           exact
written form vs target string        →  whether normalisation is needed  exact
```

No annotated real dataset provides the first two. FUNSD, CORD and SROIE annotate
*values*; on them, grounding can only be approximated by string matching, and an
approximate oracle is not an oracle — a model that reads the right value from the
wrong place scores as correct. That is why the shipped data is generated, and why
`gdx.data.real` exists but contributes **no committed number**.

## Efficiency, measured rather than quoted

<!-- table:efficiency -->
| arm | batch_size | seq_len | params | mmacs | latency_ms | iqr_ms | latency_per_doc_ms | macs_per_ms | latency_reduction_vs_generative | params_reduction_vs_generative | macs_reduction_vs_generative |
|---|---|---|---|---|---|---|---|---|---|---|---|
| span_verify | 1 | 154 | 1.09e+05 | 17.3268 | 17.7259 | 4.7454 | 17.7259 | 0.9775 | 9.3063 | 1.8472 | 7.258 |
| span_verify | 8 | 154 | 1.09e+05 | 96.1157 | 99.5228 | 10.0745 | 12.4403 | 0.9658 | 3.3315 | 1.8472 | 10.0251 |
| generative | 1 | 154 | 2.01e+05 | 125.759 | 164.9634 | 12.2422 | 164.9634 | 0.7623 | 1 | 1 | 1 |
| generative | 8 | 154 | 2.01e+05 | 963.5727 | 331.5582 | 24.3765 | 41.4448 | 2.9062 | 1 | 1 | 1 |
<!-- /table -->

Ten warm-up iterations, thirty timed repeats, median and IQR, 2 torch threads,
154 tokens. Latency covers the whole inference path — encode, decode, and the
bounded verification loop — because timing the encoder alone would flatter both
arms equally and hide the cost of the mechanism this repository adds.

**Read the MACs-per-ms column, not just the MAC ratio.** At batch 8 the
generative head retires 2.91 MMAC/ms against selection's 0.97: its decoder is
dense matrix multiplication, while span decoding is dominated by fixed overhead
it cannot amortise. The MAC reduction is 10.03x and the latency reduction is
3.33x. Quoting the first number alone would have been misleading, which is
exactly why both are here.

**The claim that actually separates the two approaches is asymptotic.** A span
head emits two indices: `O(1)` sequential steps, independent of value length. A
character decoder emits `L` characters in `L` sequential steps *per field*. Both
exponents are fitted from measurements:

<!-- table:decode_scaling -->
| head | dec_max_len | latency_ms | iqr_ms | fitted_exponent |
|---|---|---|---|---|
| span | 4 | 82.844 | 8.1828 | 0.0776 |
| span | 8 | 80.0308 | 8.9176 | 0.0776 |
| span | 12 | 117.5483 | 29.6286 | 0.0776 |
| span | 16 | 82.6206 | 8.1945 | 0.0776 |
| span | 20 | 91.5661 | 16.9817 | 0.0776 |
| span | 24 | 96.3721 | 30.3007 | 0.0776 |
| generative | 4 | 92.154 | 10.6011 | 0.7551 |
| generative | 8 | 175.145 | 54.2506 | 0.7551 |
| generative | 12 | 211.1591 | 45.6685 | 0.7551 |
| generative | 16 | 259.2902 | 10.2136 | 0.7551 |
| generative | 20 | 317.8147 | 12.0141 | 0.7551 |
| generative | 24 | 377.5217 | 31.2828 | 0.7551 |
<!-- /table -->

Selection's 0.078 is flat to within its own IQR. Generation's 0.755 falls short
of 1.0 because each decode also pays a fixed encoder pass; the *marginal* cost per
character is linear, as the raw latencies show (92 ms at 4 characters, 378 ms at
24).

**The other cost axis is tokens, and it is accounting rather than latency:**

<!-- table:baseline_cost -->
| arm | metric | value | n_docs | note |
|---|---|---|---|---|
| heuristic | latency_per_doc_ms | 7.5227 | 40 | variant row_first, warmup 2 |
| llm_stub | prompt_tokens_per_doc | 1119.4 | 40 | 8 calls/doc, whitespace tokens, lower bound |
| span_verify | prompt_tokens_per_doc | 0 | 40 | no provider call; local model |
<!-- /table -->

An LLM extraction pipeline pays for the whole serialised document on every
request, once per field. The count is whitespace-based, so it is a **lower bound**
on what a real provider would bill. The `llm_stub` accuracy numbers elsewhere come
from a deterministic offline stub and are **not** a measurement of any language
model — see [docs/METHOD.md §6](docs/METHOD.md#6-the-llm-arm).

## Results

Seven arms, one dataset, one training loop, three seeds; 1200 training documents,
400 test documents, 8 fields, 3200 (document, field) pairs per seed. The primary
seed is 0.

<!-- table:method -->
| arm | family | strict_accuracy | canonical_accuracy | coverage | hallucination_rate | grounding_exact | grounding_iou | f1 | absent_abstain_rate | n_records |
|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | baseline | 0.7625 | 0.8269 | 0.755 | 0 | 0.9652 | 0.9671 | 0.8797 | 1 | 3200 |
| llm_stub | baseline | 0.7566 | 0.7566 | 0.7459 | 0 | n/a | n/a | 0.7992 | 1 | 3200 |
| generative | reference | 0.0494 | 0.0494 | 0.8909 | 0.9986 | n/a | n/a | 6.97e-04 | 0.4968 | 3200 |
| generative_verify | baseline | 0.0988 | 0.0988 | 0.0013 | 0 | n/a | n/a | 0.0014 | 1 | 3200 |
| span_only | ablation | 0.8759 | 0.9394 | 0.9163 | 0 | 0.9623 | 0.9651 | 0.9481 | 0.7898 | 3200 |
| span_verify | ours | 0.8825 | **0.9487** | 0.8641 | 0 | 0.9909 | 0.9911 | 0.9666 | 0.9713 | 3200 |
| span_verify_norm | ours | 0.9481 | **0.9487** | 0.8641 | 0 | 0.9909 | 0.9911 | 0.9666 | 0.9713 | 3200 |
<!-- /table -->

`heuristic` and both `span` arms select, so their hallucination rate is 0 by
construction. `llm_stub` and `generative` generate. `generative_verify` generates
*and then checks*, which is the arm a fair reading of the argument demands.

**Read the ranking against the noise, not down the column.** A three-seed study
of the identical configuration gives the run-to-run scale of a difference:

<!-- table:verdicts -->
| arm | metric | value | reference_value | delta | noise_scale | ratio_to_noise | verdict |
|---|---|---|---|---|---|---|---|
| generative_verify | strict_accuracy | 0.0923 | 0.0477 | 0.0446 | 0.008 | 5.6016 | robust |
| heuristic | strict_accuracy | 0.7593 | 0.0477 | 0.7116 | 0.0065 | 109.2156 | robust |
| llm_stub | strict_accuracy | 0.7574 | 0.0477 | 0.7097 | 0.0065 | 108.9279 | robust |
| span_only | strict_accuracy | 0.8818 | 0.0477 | 0.8341 | 0.0135 | 61.6434 | robust |
| span_verify | strict_accuracy | 0.8887 | 0.0477 | 0.841 | 0.0127 | 66.096 | robust |
| span_verify_norm | strict_accuracy | 0.9541 | 0.0477 | 0.9064 | 0.0134 | 67.5777 | robust |
| generative_verify | canonical_accuracy | 0.0923 | 0.0477 | 0.0446 | 0.008 | 5.6016 | robust |
| heuristic | canonical_accuracy | 0.8244 | 0.0477 | 0.7767 | 0.0065 | 119.2083 | robust |
| llm_stub | canonical_accuracy | 0.7574 | 0.0477 | 0.7097 | 0.0065 | 108.9279 | robust |
| span_only | canonical_accuracy | 0.9444 | 0.0477 | 0.8967 | 0.0122 | 73.2125 | robust |
| span_verify | canonical_accuracy | 0.9546 | 0.0477 | 0.9069 | 0.0139 | 65.2066 | robust |
| span_verify_norm | canonical_accuracy | 0.9546 | 0.0477 | 0.9069 | 0.0139 | 65.2066 | robust |
| generative_verify | coverage | 0.0013 | 0.8852 | -0.884 | 0.0133 | 66.5861 | robust |
| heuristic | coverage | 0.7576 | 0.8852 | -0.1276 | 0.0133 | 9.6121 | robust |
| llm_stub | coverage | 0.75 | 0.8852 | -0.1352 | 0.0133 | 10.1849 | robust |
| span_only | coverage | 0.9216 | 0.8852 | 0.0364 | 0.0133 | 2.7385 | survives |
| span_verify | coverage | 0.8772 | 0.8852 | -0.008 | 0.0195 | 0.4112 | inside noise |
| span_verify_norm | coverage | 0.8772 | 0.8852 | -0.008 | 0.0195 | 0.4112 | inside noise |
| generative_verify | grounding_exact | n/a | n/a | n/a | n/a | n/a | unknown |
| heuristic | grounding_exact | 0.9671 | n/a | n/a | 0.0043 | n/a | unknown |
| llm_stub | grounding_exact | n/a | n/a | n/a | n/a | n/a | unknown |
| span_only | grounding_exact | 0.9666 | n/a | n/a | 0.0062 | n/a | unknown |
| span_verify | grounding_exact | 0.9927 | n/a | n/a | 0.0022 | n/a | unknown |
| span_verify_norm | grounding_exact | 0.9927 | n/a | n/a | 0.0022 | n/a | unknown |
| generative_verify | grounding_iou | n/a | n/a | n/a | n/a | n/a | unknown |
| heuristic | grounding_iou | 0.969 | n/a | n/a | 0.004 | n/a | unknown |
| llm_stub | grounding_iou | n/a | n/a | n/a | n/a | n/a | unknown |
| span_only | grounding_iou | 0.9697 | n/a | n/a | 0.007 | n/a | unknown |
| span_verify | grounding_iou | 0.9928 | n/a | n/a | 0.0022 | n/a | unknown |
| span_verify_norm | grounding_iou | 0.9928 | n/a | n/a | 0.0022 | n/a | unknown |
<!-- /table -->

### What survives

* **The no-hallucination guarantee**, over 1075037 opportunities and 0 taken —
  1064592 enumerated spans, 1598 untrained-model emissions, 8847 trained-model
  emissions, all with verification off. This is arithmetic about the output
  space, not a statistic: `results/tables/invariant.csv` carries the denominators
  and `tests/test_invariant_no_hallucination.py` asserts the property over all
  admissible spans rather than over the ones a model happened to pick.
* **Selection beats the rule baseline**, robustly: +0.130208 canonical accuracy at
  9.362311x noise, +0.129479 strict at 10.175543x, +0.119583 coverage at
  6.130715x.
* **Selection beats the generative reference approach** on every accuracy metric,
  far outside noise — but see the caveat below about *how far*.
* **The verification loop improves grounding**: +0.026098 exact-span accuracy at
  4.207149x noise, +0.023117 box IoU at 3.287440x.
* **Verification makes abstention work.** Correct abstention on genuinely-absent
  fields rises from 0.789809 to 0.971338.
* **The span head's confidence knows when it is wrong**: error-detection AUROC
  0.905755, against 0.670352 for the rule baseline.
* **Determinism.** Two invocations agree to 0.0 on every per-item score and every
  emitted string.
* **The efficiency measurements**, which are measurements rather than inferences.

### What does not

* **"Selection prevents hallucination" as a claim about *heads*.**
  `generative_verify` reaches the same zero with the same checker, at 0.001250
  coverage. The mechanism is the provenance *check*.
* **The verification loop as an accuracy mechanism.** +0.010208 canonical accuracy
  is 0.734005x the noise scale — **inside noise** — and it costs 0.044375
  coverage at 2.274987x.
* **Any accuracy on fields requiring normalisation.** 0.000000 strict, bounded by
  construction, for every selection-based arm including the rule baseline.
* **The *size* of the gap over the generative baseline.** At 0.047708 canonical
  it is undertrained for exact-string reproduction at this capacity and budget;
  the ranking is not in doubt, the magnitude is not a stable estimate.
* **The learned head's grounding advantage over rules.** `span_only` versus
  `heuristic` on exact-span grounding is 0.085287x noise — inside noise. Without
  the verification loop, the neural head is not better at looking in the right
  place than a geometric rule.
* **Calibration of the verified arm.** `span_verify`'s ECE is 0.069292 against
  `span_only`'s 0.024006: filtering by a check makes the arm more accurate and
  *less* calibrated.
* **Anything about real documents.** This is a synthetic benchmark. It validates a
  mechanism and says nothing about how these methods rank on real scanned invoices
  with real OCR errors.

Retractions and negative results in full:
**[docs/RESULTS.md §9](docs/RESULTS.md#9-retractions-and-negative-results)**.

## Quickstart

Runs end to end with **no dataset download, no API keys, no network**.

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

```bash
# 1. Look at the data and its provenance ground truth
python -m gdx.cli data --doc-id 3

# 2. Prove it works end to end (~35 s; real training, real metrics)
python scripts/train.py --config configs/smoke.yaml

# 3. Cost: params, MACs, measured latency, decode scaling, token accounting
python scripts/benchmark_efficiency.py

# 4. The whole experiment matrix (about 2.5 hours on 2 shared threads)
python scripts/run_all.py

# 5. Tests
python -m pytest tests -q -m "not slow"
```

On Linux/macOS the interpreter is `./.venv/bin/python` and `make setup`,
`make smoke`, `make bench`, `make all`, `make test` do the same things.

> **Python 3.12, not 3.13/3.14.** The 3.14 torch wheels in this environment fail
> to import on a missing bundled `torchgen`, and the PyPI `torchgen` package is an
> unrelated stub that does not fix it.

## The data, and why it is generated

`gdx.data` is a procedural generator of visually-rich invoices. It places every
token, so it knows the exact token index range each field's value was written at,
the exact box that range covers, and whether the written form literally equals the
target.

The difficulty is not incidental. Each of these is a config switch so its
contribution can be measured rather than assumed:

| knob | what it breaks |
|---|---|
| `distractor_prob` | "Balance Forward", "Amount Paid", "Shipping" carry amounts next to the target; a bottom-right-most-number rule picks one |
| `duplicate_total_prob` | "Amount Due" reprints the total under a label that is *also* a synonym for `total` |
| `missing_field_prob` | a dropped field makes abstention correct and emitting anything an error |
| `multi_page_prob` | the summary block lands on the last page, far from the header |
| `box_jitter`, `rotation_deg` | row membership stops being an equality on `y` |
| `verbose_date_prob` | the normalisation case selection structurally cannot win |

Three details that decide whether the benchmark measures anything:

**Amount arithmetic is done in integer cents**, so `subtotal + tax = total` is
exact in the data and a violation found at inference is a real inconsistency
rather than a rounding artefact.
→ `test_arithmetic_relation_is_exact_in_the_data`

**Duplicated values count as correctly grounded from any occurrence.** The
occurrences are found by scanning the *finished* document, which catches the
coincidences that bookkeeping during layout would miss — a line-item amount that
happens to equal the tax, for instance.
→ `test_duplicate_total_creates_a_second_occurrence`

**An absent field's correct output is abstention**, scored as correct only when
the field is genuinely absent. A method cannot buy accuracy by declining.
→ `test_abstention_is_correct_exactly_when_the_field_is_absent`

> **The limitation, stated plainly.** This is synthetic. Reading order here is the
> order the generator wrote in; real OCR order is noisier, and real scans have
> recognition errors this benchmark does not model. The optional loader
> (`scripts/download_real.py`) exists so a reader can run the method on FUNSD,
> CORD or SROIE — but none of them records *where* a value was read from, so the
> grounding oracle degrades to string matching there and no committed number uses
> that path.

## The method in three pieces

**A layout-aware encoder, hand-rolled.** Token identity (closed lexicon plus a
crc32 hash tail), surface shape class, numeric magnitude bucket, raw geometry,
reading-order sinusoids, and 2-D layout Fourier features — plus a *relative*
spatial attention bias, a 332-parameter table over signed log-scale buckets of
`(Δx, Δy)` and a same-page flag. That relative term is the part absolute encodings
cannot supply: "the value is to the right of its label" is a relation, not a
position. No pretrained checkpoint, no `transformers`, no `torch-geometric` — the
claim under test is about the output interface, so the backbone is held fixed and
fully visible.

**A span-selection head whose guarantee is stated precisely.** Per-field start and
end distributions over tokens, with an admissibility mask
`s ≤ e`, `e − s + 1 ≤ L`, **and `page(s) = page(e)`**. That last clause is
load-bearing: without it a span can splice the end of one page onto the start of
the next, producing a value that is a sequence of document tokens but not a
contiguous region — and the invariant test caught exactly that, on 3 of 8 seeds,
before the clause existed. **A guarantee about an output space is only as good as
the agreement between the decoder's mask and the checker's definition.**

**A bounded verification loop.** Provenance, type, and `subtotal + tax = total`
on three separately-grounded spans. On failure the loop advances the
least-confident participating field to its next candidate, at most
`verify.max_iters = 3` times, then abstains. The bound is not cosmetic: an
unbounded repair loop is not an algorithm with a runtime, and the realised
iteration count is recorded per document.

Full derivations and design reasoning: **[docs/METHOD.md](docs/METHOD.md)**.

## What makes the comparison trustworthy

**One training loop, one encoder, one seed.** `model.head` is the *only*
configuration difference between the method and its reference baseline —
enforced by construction, not by convention, because separate scripts per arm are
how an incidental difference in schedule gets reported as a difference in method.
Both heads are built after re-seeding, so a given seed gives both the same encoder
initialisation.

**Checkpoint selection on validation loss, not on a metric.** The two heads'
natural metrics are not comparable, so selecting on a metric would apply a
different rule to each arm.

**The generative baseline is given a fair fight.** It attends over the encoder
states rather than a pooled vector (a pooled-only decoder could not read a
specific number off the page at all), it shares one decoder across fields rather
than getting eight, and it can abstain by emitting the empty string — the same
affordance the span head has.

**An arm built to refute the central claim is reported.**
`generative_verify` applies the same verification loop to generated strings, and
it *does* reach zero hallucination. That result narrows this repository's claim,
and it is in the README rather than in a footnote.

**The rule baseline is swept, on validation.** Six geometric configurations
(same-row only, row-first, below-label cheap, wide, any-page, last-label-wins),
each with all label synonyms and type-aware value matching, chosen on the
validation split and never on test. A quiet bug in its synonym matcher once made
it find labels on 20% of fields instead of 96%; reporting *that* as the rule
baseline's ceiling would have been a strawman.

**Every ablation flips one config field**, and its delta is divided by the
run-to-run noise scale before it is interpreted. Inference-time ablations reuse
the *same trained weights*, so their deltas contain no initialisation noise.

**Every number in this file is injected from a CSV** by
`scripts/render_docs.py`, and a test fails if any of them has drifted.

## Repository layout

```
configs/            one YAML per experiment, composed through `_base_`
  base.yaml           shared defaults; smoke.yaml runs in under a minute
  span / generative   the method and its reference baseline
  no_2d_pos / no_spatial_bias / no_verify   one switch each
  real_funsd.yaml     optional real-data path (no committed number)

src/gdx/
  config.py           typed config + `_base_` inheritance + --set overrides
  extract.py          the Extraction record: value, span, confidence, grounded
  verify.py           the bounded verification loop and its three checks
  arms.py             the seven comparison arms
  data/
    schema.py           Document, FieldTruth (value + provenance), normalisers
    lexicon.py          label synonyms, distractor labels, currency styles
    layout.py           box placement, page noise, IoU
    generator.py        the invoice generator and its exact ground truth
    featurise.py        vocabulary, shape classes, magnitude buckets, geometry
    dataset.py          splits, the [CLS] null convention, batching
    real.py             optional FUNSD / CORD / SROIE loader
  models/
    position.py         reading-order sinusoids, 2-D Fourier, spatial bias
    attention.py        hand-rolled multi-head attention with an additive bias
    encoder.py          the layout encoder
    heads.py            SpanHead (selection) and GenerativeHead (the reference)
    model.py            the assembled model; `head` is the only difference
  baselines/
    heuristic.py        label synonyms + geometric search, six swept variants
    llm.py              the LLM-pipeline arm
  llm/                  LLMClient, HTTP providers, the deterministic stub
  metrics/
    fields.py  grounding.py  calibration.py  stats.py
  engine/trainer.py   ONE loop for both heads
  pipelines/          core, experiments, analysis, ablations, efficiency,
                      invariant, determinism
  report.py  viz.py  cli.py  utils/

scripts/            train, run_experiments, run_ablations, benchmark_efficiency,
                    analyse, make_figures, render_docs, build_notebooks,
                    download_real, run_all
notebooks/          01 data & provenance · 02 selection vs generation
                    03 ablations & noise · 04 abstention & calibration
                    05 Colab full scale
tests/              643 tests
docs/               METHOD.md · RESULTS.md · REPRODUCIBILITY.md
results/            tables/ figures/ runs/ — the evidence, committed
```

## Notebooks

| notebook | what it establishes | needs a GPU |
|---|---|---|
| `01_data_and_provenance.ipynb` | the ground truth is provenance, not strings; the invariant holds on data alone; the page-splice counterexample | no |
| `02_selection_vs_generation.ipynb` | both heads on one encoder, and what the generative head actually emits next to the truth | no |
| `03_ablations_and_noise.ipynb` | which switch does the work, against the noise floor, and the measured cost of the normalisation limitation | no |
| `04_abstention_and_calibration.ipynb` | when declining is correct, whether confidence tracks correctness, and what the loop did | no |
| `05_colab_full_scale.ipynb` | the same code at ~8x scale, plus the optional real-data path | yes |

## Configuration

One YAML fully describes an experiment, every run saves its *resolved* config next
to its results, and **unknown keys are rejected** — a silently-swallowed typo
would void an experiment while producing a plausible number.

```bash
python scripts/train.py --config configs/span.yaml \
  --set optim.epochs=12 verify.arithmetic=false data.verbose_date_prob=0.0
```

## Citation

```bibtex
@software{ahmad2026gdx,
  author = {Arslan Ahmad},
  title  = {Where Did That Number Come From? Provenance-Grounded Document
            Extraction with a Bounded Verification Loop},
  year   = {2026},
  url    = {https://github.com/arslan-ahmad/grounded-document-extraction}
}
```

Builds on: Devlin et al. BERT extractive span heads (2019); Bahdanau et al.
attention (2015); Raffel et al. T5 relative-position bias (2020); Tancik et al.
Fourier features (2020); Xu et al. LayoutLM (2020); Kim et al. Donut (2022);
Biten et al. ANLS (2019); Guo et al. calibration (2017); Nixon et al. adaptive
calibration error (2019); Geifman & El-Yaniv selective classification (2017);
Holm (1979); Wilcoxon (1945). Full list in
[docs/METHOD.md §8](docs/METHOD.md#8-references).

## License

MIT — see [LICENSE](LICENSE).
