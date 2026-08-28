# Results

Every table in this file is injected from a CSV in `results/tables/` by
`scripts/render_docs.py`, and every number quoted in the prose is checked against
those CSVs by `tests/test_doc_numbers.py`. Nothing here is typed from memory.

> **The four sentences that matter.**
>
> 1. **The guarantee holds and is exact.** Zero ungrounded values from the
>    span arms across 3 seeds and 9,600 (document, field) pairs, against a
>    hallucination rate of 0.9986 for the generative reference approach on the
>    same encoder, same data, same budget, same seed.
> 2. **But the checker, not the head, is what removes them.** Applying the same
>    provenance check to the generative arm also reaches 0.0000 — by abstaining on
>    99.9% of fields. Selection's real contribution is that it satisfies the check
>    *for free*, at 0.8772 coverage instead of 0.0013.
> 3. **Selection beats the strong rule baseline**, +0.1302 canonical accuracy at
>    9.36x the run-to-run noise scale, and beats it on grounding by +0.0256 at
>    5.92x — but that grounding gain comes from the verification loop, not from
>    the head: `span_only` versus `heuristic` on grounding is 0.085x noise,
>    inside noise.
> 4. **The ideology's cost is exactly zero accuracy where it applies.** On the
>    211 test fields whose written form is not the target, every selection-based
>    arm scores **0.000000** strict accuracy — this repository's method and the
>    rule baseline alike. It is a property of selection, not of this model.

## What was run, and what was not

**Run.** Seven arms, three seeds, one dataset population per seed, one training
loop. Per seed: 1200 train / 300 validation / 400 test documents, 8 fields, so
3200 (document, field) pairs per arm per seed and 9600 per arm in total. Both
heads trained per seed; the rule baseline swept over six geometric configurations
and selected on validation; the LLM arm answered 8 prompts per document from the
offline stub. Inference-time ablations reuse one span checkpoint per seed;
training-time ablations retrain. Latency, MACs, decode scaling and token cost
measured separately with 10 warm-up and 30 timed repeats.

**Not run.** No real dataset. No pretrained checkpoint. No beam search for the
generative head. No hyper-parameter search for the neural arms beyond the shared
defaults — the compute budget went to seeds and ablations instead, which is the
right allocation when the question is whether a difference exceeds noise.

**The scale, stated so it is not mistaken for something larger.** A 64-wide
2-layer encoder, 1200 training documents, 6 epochs, three seeds, on two CPU
threads. The structural claims are scale-free. The accuracy comparisons are the
ones a larger study could move, and §9 says which.

## 1. Efficiency

<!-- table:efficiency -->
<!-- /table -->

Ten warm-up iterations, thirty timed repeats, median and IQR, 2 torch threads,
154 tokens. Latency covers the whole inference path — encode, decode, and the
bounded verification loop — because timing the encoder alone would flatter both
arms equally and hide the cost of the mechanism this repository adds.

At batch 1 the selection path runs in 17.72595 ms against 164.9634 ms for
generation, a factor of **9.306322**. At batch 8 the factor falls to **3.33148**,
because the generative decoder's sequential steps amortise across the batch while
selection's fixed overhead does not.

**Read the MACs-per-ms column, not the MAC ratio.** At batch 8 generation retires
2.906195 MMAC/ms against selection's 0.965766 — its decoder is dense matrix
multiplication, while span decoding is dominated by overhead it cannot amortise.
The MAC reduction there is 10.025132x and the latency reduction is 3.33148x.
Quoting the first number alone would have been misleading, which is why both are
in the table.

### 1.1 The claim that actually separates the two approaches is asymptotic

<!-- table:decode_scaling -->
<!-- /table -->

A span head emits two indices: a fixed number of sequential steps, independent of
how long the value is. A character decoder emits `L` characters in `L` sequential
steps *per field*. Fitted exponents in value length: **0.077604** for selection
and **0.755079** for generation.

Two honest qualifications. Selection's 0.077604 is flat only to within its own
noise — its latencies wander between 80.0308 ms and 117.54825 ms with IQRs as
large as 30.300675 ms, so the fit is "no detectable trend" rather than a precise
zero. Generation's 0.755079 falls short of 1.0 because each decode also pays a
fixed encoder pass that does not grow with `L`; the marginal cost per character is
what is linear, and the raw latencies show it (92.154 ms at 4 characters,
377.52165 ms at 24).

### 1.2 The other cost axis is tokens, and it is accounting rather than latency

<!-- table:baseline_cost -->
<!-- /table -->

An LLM extraction pipeline pays for the whole serialised document on every
request, once per field. The count is whitespace-based, so it understates a real
BPE tokeniser on numeric text and is a **lower bound** on what a provider would
bill. The rule baseline's per-document latency is measured under the same
protocol as the neural arms.

## 2. The seven-arm comparison

<!-- table:method -->
<!-- /table -->

Seed 0. `heuristic` and both `span` arms select, so their hallucination rate is 0
by construction. `llm_stub` and `generative` generate. `generative_verify`
generates *and then checks*, which is the arm a fair reading of the argument
demands.

Three things to read off this table.

**The reference approach is not competitive at this budget.** `generative`
reaches 0.049375 canonical accuracy. That is not "generation cannot extract" — it
is "a character decoder at this capacity and budget cannot reproduce exact
strings", and §9.3 gives the evidence for which of the two it is.

**`llm_stub` is the honest surprise.** At 0.756563 canonical it is within noise of
the rule baseline on strict accuracy (0.467572x the noise scale, inside noise)
despite having no geometry at all. Reading a linearised token stream with
last-label-wins gets most invoice fields right. It loses on the fields where
layout disambiguates — see §2.1.

**`span_verify` trades coverage for precision.** 0.987703 precision at 0.864062
coverage, against `span_only`'s 0.940655 at 0.916250. The verification loop is
doing what it is for.

### 2.1 Per field

<!-- table:per_field -->
<!-- /table -->

The fields where selection is near-perfect are the ones with an unambiguous
anchor: `vendor_name` 1.0 strict, `invoice_id` 0.9975. The hard ones are the
amounts, where the distractor lines live: `total` 0.8675 strict at 0.8725
coverage. `invoice_date` shows the normalisation limitation directly — 0.68
strict against 0.9775 canonical, the gap being exactly the non-ISO dates.
`po_number`'s 0.555 coverage is not a failure: it is absent from 43% of documents
and abstention is correct there.

### 2.2 Is the difference significant?

<!-- table:statistical_tests -->
<!-- /table -->

Paired tests on seed 0, Holm-corrected across the metric family. The unit column
matters: `document` means per-field values were averaged within a document first,
because the eight fields of one invoice share a layout and are not independent;
`set` means the statistic has no per-item value and a paired bootstrap over
documents was used instead.

**The caveat that applies to all of them.** These condition on **one trained model
per arm**. They are statements about two sets of weights, not about two methods.
The method-level question is answered by the seed study below, and where the two
disagree the seed study wins.

### 2.3 And is it bigger than the noise?

<!-- table:seed_variance -->
<!-- /table -->

Three seeds of the identical configuration, varying both the initialisation and
the generated population — which is the honest thing to vary, because reusing one
dataset across seeds would report initialisation noise only and understate the
run-to-run scale.

<!-- table:verdicts -->
<!-- /table -->

Against the generative reference approach, every arm's accuracy gap is far outside
noise — but the ratios (up to 139.11365x) are large mostly because `generative` is
both very low and very stable, so its noise scale is only 0.006515. A ratio that
size is not a more confident claim than one at 10x; it is a claim about a
comparison whose denominator is small. The grounding rows come back `unknown`
because generation makes no span claim at all: there is nothing to compare.

The more informative comparison is against the rule baseline, which is the
stronger competitor:

<!-- table:verdicts_vs_heuristic -->
<!-- /table -->

* `span_verify` over `heuristic`: **+0.130208 canonical at 9.362311x noise**,
  **+0.129479 strict at 10.175543x**, **+0.119583 coverage at 6.130715x**. All
  robust.
* `span_verify_norm` over `heuristic` on strict accuracy: **+0.194792 at
  14.523642x**.
* **Grounding is where the story turns.** `span_verify` beats `heuristic` by
  +0.025569 at 5.924370x — but `span_only` versus `heuristic` is −0.000529, a
  ratio of 0.085287, **inside noise**. The learned head is not better at *looking
  in the right place* than the rule baseline; the verification loop is.
* `llm_stub` versus `heuristic` on strict accuracy: 0.467572x, **inside noise**.

## 3. The hallucination result, stated precisely

Per-seed hallucination rate — the fraction of *emitted* values that occur nowhere
in the document, denominator being emissions rather than pairs, so an arm that
abstains everywhere earns nothing:

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| `generative` | 0.998597 | 0.999285 | 0.997894 |
| `span_only`, `span_verify`, `span_verify_norm` | 0.0 | 0.0 | 0.0 |
| `heuristic`, `llm_stub`, `generative_verify` | 0.0 | 0.0 | 0.0 |

Two of those zeros mean different things, and conflating them would be the single
easiest mistake to make here.

**Structural zeros.** `heuristic`, `span_only`, `span_verify` and
`span_verify_norm` *select*. Whatever they emit is a rendering of a token span,
so the string is in the document by construction. `span_only` reaches zero with
the verification loop **switched off**, which is the point: nothing is checking
it, and it still cannot fabricate.

**A checked zero.** `generative_verify` reaches zero by *refusing*. Its coverage
is 0.001250 against `generative`'s 0.890938 — it emits 4 values on seed 0 where
the unchecked arm emits 2851, and it spends 7.1175 verification iterations per
document doing so. The provenance check works perfectly on generated strings; it
just has almost nothing to accept.

**So the claim this repository can support is narrower than "selection prevents
hallucination".** The provenance *check* is what removes ungrounded values, from
either head. Selection's contribution is that it passes the check for free — at
0.877188 coverage (3-seed mean) rather than 0.001250 — and that it needs no
checker at all to be safe. That distinction only became visible because the arm
capable of refuting the broad claim was built and reported.

### 3.1 The guarantee, counted

<!-- table:invariant -->
<!-- /table -->

The table quantifies over three independent denominators: every admissible span
of every document (a statement about the output *space*, not about any model),
the emissions of untrained networks across eight seeds (so the property cannot be
an artefact of training), and the emissions of trained checkpoints
(what a deployment would see). Verification is **disabled** for the model rows on
purpose — with the checker on, a zero would be unremarkable.

`tests/test_invariant_no_hallucination.py` asserts the same property as a test,
over 8 seeds, including a deliberately-injected absent string to prove the
detector fires and an adversarial-span case (inverted, out-of-range, cross-page)
to prove no malformed selection escapes as a value.

## 4. The measured cost of the ideology

<!-- table:normalisation_cost -->
<!-- /table -->

With probability 0.30 the generator writes a date in a form that is not the
target: `03/01/2024`, `Jan 3, 2024`, `3rd of Jan 2024` against a target of
`2024-01-03`. No contiguous span of document tokens equals the target, so a
selection-only extractor **cannot** be right there.

On seed 0 that is 211 of the 2886 present fields, and the measured strict accuracy
on them is:

* `span_verify` — **0.000000**
* `span_only` — **0.000000**
* `heuristic` — **0.000000**

Three independent selection-based implementations, all exactly zero. This is not
a model failure; it is the ideology's cost, and it is why the number is stated
here rather than avoided by evaluating only on verbatim fields, where the same
arms score 0.941682, 0.955140 and 0.794766.

Two things soften it and one thing does not.

**Canonical accuracy is fine.** `span_verify` reaches 0.995261 on those fields
under a comparison that parses dates. If the downstream consumer can parse, the
value was read correctly and the limitation is cosmetic.

**A deterministic post-processor recovers it.** `span_verify_norm` scores
0.995261 strict on the same fields — the highest of any arm, including
`llm_stub`'s 0.729858. But it **weakens the guarantee**: after normalisation the
emitted string is no longer a document substring, only a pure deterministic
function of a grounded span. Provenance survives; property (P1) does not. That is
stated wherever this arm appears.

**What does not soften it:** if the target format is fixed and no post-processor
exists for it, selection is simply unable to produce it, and no amount of training
changes that. A generative head has no such bound — it merely failed to exploit it
here (0.000000 strict on the same fields, because at this budget it cannot spell).

## 5. Abstention and calibration

<!-- table:abstention -->
<!-- /table -->

`absent_abstain_rate` is the one that matters: on the 314 fields the generator
genuinely omitted, abstaining is the *correct* output. `span_verify` reaches
0.971338 there against `span_only`'s 0.789809 — the verification loop's clearest
single win. `heuristic` and `llm_stub` reach 1.000000, but by a weaker route:
they emit nothing when no label is found, and a missing field usually has no
label.

The cost is on the other side. `present_emit_rate` — the fraction of genuinely
present fields the arm answers — is 0.954955 for `span_verify` against 0.993070
for `span_only`. The loop declines on about 4% of answerable fields to gain 18
points of correct-abstention.

The loop's own cost is small: on seed 0 it runs 0.6250 iterations per document
with 0.5125 re-selections. `generative_verify` burns 7.1175 iterations per
document by comparison, because almost every candidate it is offered fails.

<!-- table:calibration -->
<!-- /table -->

Measured on emitted values only: an abstention's confidence is a confidence in
*absence*, and pooling the two would produce a reliability diagram about two
different questions.

* `span_only` is the best-calibrated arm — ECE 0.024006, ACE 0.025422, MCE
  0.099913 — and `span_verify` is *worse* on ECE (0.069292) despite being more
  accurate. Filtering by a check removes low-confidence *errors* and leaves the
  surviving confidences systematically too low relative to a now-higher accuracy.
  Verification improves accuracy and damages calibration, and both are reported.
* Error-detection AUROC is 0.905755 for `span_verify` and 0.905418 for
  `span_only` — the span head's confidence carries real information about which
  answers are wrong. The rule baseline's 0.670352 is much weaker, as expected from
  a rank heuristic dressed as a probability.
* AURC is 0.001377 for `span_verify` against 0.020874 for `heuristic` and
  0.998286 for `generative`.
* **`generative_verify`'s calibration row is meaningless and is kept only for
  completeness**: it is computed over 4 emitted values. `llm_stub`'s row is all
  `n/a` because a text completion yields no confidence, and the arm reports `NaN`
  rather than a fabricated 1.0.

## 6. The rule baseline, given a fair fight

A weak rule baseline would make every neural comparison in this file worthless,
so its geometry is swept and the winner chosen on **validation**, never on test:

<!-- table:heuristic_sweep -->
<!-- /table -->

Six configurations over 150 validation documents (1200 pairs). `row_only` and
`row_first` tie at 0.826667 canonical accuracy and the sweep takes `row_only`;
they differ sharply elsewhere, with `row_only` at 0.976898 grounding exactness
against `row_first`'s 0.915464 — the looser variant finds more values and puts
more of them in the wrong place. The worst configurations (`row_first_wide`,
`any_page`) reach 0.774167, so the sweep is worth 0.052500 canonical accuracy on
its own.

The baseline also gets: every label synonym the generator can emit, from the
*same* shared table the generator draws from; multi-word label matching with
longest-synonym-first, so "Total Due" wins over the "Total" inside it; and
type-aware value matching, so it never proposes a word where an amount belongs.

**And it inherits the grounding guarantee**, because it selects. Its
hallucination rate is 0.0 for the same structural reason this repository's method
has one. That is worth saying plainly: **the no-hallucination property belongs to
selection, not to neural networks**, and the comparison this project cares about
is selection versus generation rather than learned versus rule-based.

One bug found here is worth recording. The synonym matcher originally broke out of
its loop on the first *non*-matching synonym, so only the longest synonym was ever
tried and the baseline located labels on about 20% of fields instead of 96%.
Reporting that as the rule baseline's ceiling would have been a strawman, and the
regression test `test_heuristic_finds_most_labels_on_real_documents` now pins it.

## 7. Ablations

<!-- table:ablations -->
<!-- /table -->

Two families, and keeping them apart matters. **Inference-time** switches reuse
the *same trained weights* and the same decoded candidate sets, so their deltas
contain no initialisation noise at all — the noise scale they are divided by is
the `full` arm's seed-to-seed spread, which asks whether the switch's effect is
larger than the variability of the pipeline it sits inside. **Training-time**
switches retrain, so their deltas carry run-to-run noise directly.

### 7.1 The verification loop, isolated

`span_only` is the same weights and the same decoded candidates with the loop
switched off, so comparing against it attributes the loop exactly:

<!-- table:verdicts_vs_span_only -->
<!-- /table -->

Read down the `verdict` column, not the `delta` column.

* **Grounding: robust.** +0.026098 exact-span accuracy at 4.207149x noise, and
  +0.023117 box IoU at 3.287440x. The loop reliably moves the *location* the value
  was read from onto the right span.
* **Coverage: a real cost, and it survives.** −0.044375 at 2.274987x. This is not
  a rounding effect; the loop declines to answer.
* **Accuracy: inside noise.** +0.010208 canonical at 0.734005x, +0.006979 strict
  at 0.515812x. **The verification loop does not demonstrably improve accuracy at
  this scale.** It improves *precision* (0.987703 against 0.940655 on seed 0) and
  correct abstention (0.971338 against 0.789809), and it pays for both in
  coverage; the net effect on accuracy over all pairs is not distinguishable from
  reseeding.
* `span_verify_norm`'s +0.072292 strict at 5.342889x is robust, but that is the
  date normaliser, not the loop — the two arms share a verification configuration
  and differ only in the post-processor.
