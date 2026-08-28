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
