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
threads of a shared machine. The structural claims are scale-free. The accuracy comparisons are the
ones a larger study could move, and §9 says which.

## 1. Efficiency

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
| arm | metric | value | n_docs | note |
|---|---|---|---|---|
| heuristic | latency_per_doc_ms | 7.5227 | 40 | variant row_first, warmup 2 |
| llm_stub | prompt_tokens_per_doc | 1119.4 | 40 | 8 calls/doc, whitespace tokens, lower bound |
| span_verify | prompt_tokens_per_doc | 0 | 40 | no provider call; local model |
<!-- /table -->

An LLM extraction pipeline pays for the whole serialised document on every
request, once per field. The count is whitespace-based, so it understates a real
BPE tokeniser on numeric text and is a **lower bound** on what a provider would
bill. The rule baseline's per-document latency is measured under the same
protocol as the neural arms.

## 2. The seven-arm comparison

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
| arm | field | n | n_present | strict_accuracy | canonical_accuracy | coverage | grounding_exact |
|---|---|---|---|---|---|---|---|
| generative | invoice_id | 400 | 400 | 0 | 0 | 1 | 0 |
| generative | invoice_date | 400 | 400 | 0.005 | 0.005 | 1 | 0 |
| generative | due_date | 400 | 346 | 0 | 0 | 1 | 0 |
| generative | vendor_name | 400 | 400 | 0 | 0 | 1 | 0 |
| generative | po_number | 400 | 227 | 0.3875 | 0.3875 | 0.135 | 0 |
| generative | subtotal | 400 | 362 | 0 | 0 | 0.9975 | 0 |
| generative | tax | 400 | 351 | 0.0025 | 0.0025 | 0.995 | 0 |
| generative | total | 400 | 400 | 0 | 0 | 1 | 0 |
| heuristic | invoice_id | 400 | 400 | 0.9775 | 0.9775 | 0.9775 | 0.9775 |
| heuristic | invoice_date | 400 | 400 | 0.585 | 0.8525 | 0.9875 | 0.8525 |
| heuristic | due_date | 400 | 346 | 0.735 | 0.96 | 0.825 | 0.9538 |
| heuristic | vendor_name | 400 | 400 | 0 | 0 | 0.0425 | 0 |
| heuristic | po_number | 400 | 227 | 0.99 | 0.99 | 0.5575 | 0.9824 |
| heuristic | subtotal | 400 | 362 | 0.94 | 0.95 | 0.8625 | 0.9448 |
| heuristic | tax | 400 | 351 | 0.9275 | 0.9375 | 0.8275 | 0.9288 |
| heuristic | total | 400 | 400 | 0.945 | 0.9475 | 0.96 | 0.9475 |
| span_verify | invoice_id | 400 | 400 | 0.9975 | 0.9975 | 1 | 0.9975 |
| span_verify | invoice_date | 400 | 400 | 0.68 | 0.9775 | 0.995 | 0.9775 |
| span_verify | due_date | 400 | 346 | 0.7375 | 0.965 | 0.88 | 0.9769 |
| span_verify | vendor_name | 400 | 400 | 1 | 1 | 1 | 1 |
| span_verify | po_number | 400 | 227 | 0.9875 | 0.9875 | 0.555 | 0.978 |
| span_verify | subtotal | 400 | 362 | 0.895 | 0.895 | 0.825 | 0.8867 |
| span_verify | tax | 400 | 351 | 0.895 | 0.8975 | 0.785 | 0.8889 |
| span_verify | total | 400 | 400 | 0.8675 | 0.87 | 0.8725 | 0.87 |
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
| name_a | name_b | metric | unit | mean_a | mean_b | difference | ci_lower | ci_upper | p_value | p_adjusted | effect_size | n | significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | generative | hallucination_rate | set | 0 | 0.9986 | -0.9983 | -0.9996 | -0.9967 | n/a | n/a | n/a | 2000 | yes |
| llm_stub | generative | hallucination_rate | set | 0 | 0.9986 | -0.9983 | -0.9996 | -0.9962 | n/a | n/a | n/a | 2000 | yes |
| generative_verify | generative | hallucination_rate | set | 0 | 0.9986 | -1 | -1 | -1 | n/a | n/a | n/a | 2000 | yes |
| span_only | generative | hallucination_rate | set | 0 | 0.9986 | -0.9986 | -0.9996 | -0.9972 | n/a | n/a | n/a | 2000 | yes |
| span_verify | generative | hallucination_rate | set | 0 | 0.9986 | -0.9986 | -0.9996 | -0.9971 | n/a | n/a | n/a | 2000 | yes |
| span_verify_norm | generative | hallucination_rate | set | 0 | 0.9986 | -0.9986 | -0.9996 | -0.9971 | n/a | n/a | n/a | 2000 | yes |
| heuristic | generative | strict_accuracy | document | 0.7625 | 0.0494 | 0.7131 | 0.7003 | 0.7256 | 6.53e-69 | 7.83e-68 | 5.3296 | 400 | yes |
| heuristic | generative | canonical_accuracy | document | 0.8269 | 0.0494 | 0.7775 | 0.765 | 0.7891 | 6.02e-70 | 1.02e-68 | 6.3516 | 400 | yes |
| heuristic | generative | coverage | document | 0.755 | 0.8909 | -0.1359 | -0.1491 | -0.1231 | 4.69e-51 | 3.29e-50 | -1.0729 | 400 | yes |
| llm_stub | generative | strict_accuracy | document | 0.7566 | 0.0494 | 0.7072 | 0.6953 | 0.7197 | 1.85e-68 | 1.67e-67 | 5.4116 | 400 | yes |
| llm_stub | generative | canonical_accuracy | document | 0.7566 | 0.0494 | 0.7072 | 0.6953 | 0.7197 | 1.85e-68 | 1.67e-67 | 5.4116 | 400 | yes |
| llm_stub | generative | coverage | document | 0.7459 | 0.8909 | -0.145 | -0.1622 | -0.1291 | 1.98e-40 | 1.19e-39 | -0.8661 | 400 | yes |
| generative_verify | generative | strict_accuracy | document | 0.0988 | 0.0494 | 0.0494 | 0.0419 | 0.0575 | 3.81e-28 | 1.90e-27 | 0.6584 | 400 | yes |
| generative_verify | generative | canonical_accuracy | document | 0.0988 | 0.0494 | 0.0494 | 0.0419 | 0.0575 | 3.81e-28 | 1.90e-27 | 0.6584 | 400 | yes |
| generative_verify | generative | coverage | document | 0.0013 | 0.8909 | -0.8897 | -0.8944 | -0.885 | 2.08e-79 | 3.74e-78 | -18.1324 | 400 | yes |
| span_only | generative | strict_accuracy | document | 0.8759 | 0.0494 | 0.8266 | 0.8147 | 0.8391 | 7.82e-69 | 8.61e-68 | 6.5801 | 400 | yes |
| span_only | generative | canonical_accuracy | document | 0.9394 | 0.0494 | 0.89 | 0.8794 | 0.9006 | 1.43e-69 | 1.85e-68 | 8.2912 | 400 | yes |
| span_only | generative | coverage | document | 0.9163 | 0.8909 | 0.0253 | 0.0163 | 0.0344 | 5.50e-08 | 1.65e-07 | 0.2828 | 400 | yes |
| span_verify | generative | strict_accuracy | document | 0.8825 | 0.0494 | 0.8331 | 0.8197 | 0.8462 | 1.07e-68 | 1.07e-67 | 6.0571 | 400 | yes |
| span_verify | generative | canonical_accuracy | document | 0.9487 | 0.0494 | 0.8994 | 0.8878 | 0.91 | 8.70e-70 | 1.39e-68 | 7.7176 | 400 | yes |
| span_verify | generative | coverage | document | 0.8641 | 0.8909 | -0.0269 | -0.0422 | -0.0128 | 0.0013 | 0.0025 | -0.1816 | 400 | yes |
| span_verify_norm | generative | strict_accuracy | document | 0.9481 | 0.0494 | 0.8988 | 0.8869 | 0.9097 | 9.21e-70 | 1.39e-68 | 7.7037 | 400 | yes |
| span_verify_norm | generative | canonical_accuracy | document | 0.9487 | 0.0494 | 0.8994 | 0.8878 | 0.91 | 8.70e-70 | 1.39e-68 | 7.7176 | 400 | yes |
| span_verify_norm | generative | coverage | document | 0.8641 | 0.8909 | -0.0269 | -0.0422 | -0.0128 | 0.0013 | 0.0025 | -0.1816 | 400 | yes |
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
| arm | metric | n_seeds | mean | sd | noise_scale | min | max |
|---|---|---|---|---|---|---|---|
| generative | canonical_accuracy | 3 | 0.0477 | 0.0046 | 0.0065 | 0.0425 | 0.0512 |
| generative_verify | canonical_accuracy | 3 | 0.0923 | 0.0056 | 0.008 | 0.0884 | 0.0988 |
| heuristic | canonical_accuracy | 3 | 0.8244 | 0.0022 | 0.0031 | 0.8228 | 0.8269 |
| llm_stub | canonical_accuracy | 3 | 0.7574 | 0.0012 | 0.0017 | 0.7566 | 0.7588 |
| span_only | canonical_accuracy | 3 | 0.9444 | 0.0087 | 0.0122 | 0.9394 | 0.9544 |
| span_verify | canonical_accuracy | 3 | 0.9546 | 0.0098 | 0.0139 | 0.9487 | 0.9659 |
| span_verify_norm | canonical_accuracy | 3 | 0.9546 | 0.0098 | 0.0139 | 0.9487 | 0.9659 |
| generative | coverage | 3 | 0.8852 | 0.0094 | 0.0133 | 0.8744 | 0.8909 |
| generative_verify | coverage | 3 | 0.0013 | 6.25e-04 | 8.84e-04 | 6.25e-04 | 0.0019 |
| heuristic | coverage | 3 | 0.7576 | 0.0035 | 0.0049 | 0.755 | 0.7616 |
| llm_stub | coverage | 3 | 0.75 | 0.0084 | 0.0119 | 0.7444 | 0.7597 |
| span_only | coverage | 3 | 0.9216 | 0.0049 | 0.0069 | 0.9163 | 0.9259 |
| span_verify | coverage | 3 | 0.8772 | 0.0138 | 0.0195 | 0.8641 | 0.8916 |
| span_verify_norm | coverage | 3 | 0.8772 | 0.0138 | 0.0195 | 0.8641 | 0.8916 |
| generative | grounding_exact | 0 | n/a | n/a | n/a | n/a | n/a |
| generative_verify | grounding_exact | 0 | n/a | n/a | n/a | n/a | n/a |
| heuristic | grounding_exact | 3 | 0.9671 | 0.0031 | 0.0043 | 0.9652 | 0.9707 |
| llm_stub | grounding_exact | 0 | n/a | n/a | n/a | n/a | n/a |
| span_only | grounding_exact | 3 | 0.9666 | 0.0044 | 0.0062 | 0.9623 | 0.9711 |
| span_verify | grounding_exact | 3 | 0.9927 | 0.0015 | 0.0022 | 0.9909 | 0.9937 |
| span_verify_norm | grounding_exact | 3 | 0.9927 | 0.0015 | 0.0022 | 0.9909 | 0.9937 |
| generative | grounding_iou | 0 | n/a | n/a | n/a | n/a | n/a |
| generative_verify | grounding_iou | 0 | n/a | n/a | n/a | n/a | n/a |
| heuristic | grounding_iou | 3 | 0.969 | 0.0028 | 0.004 | 0.9671 | 0.9723 |
| llm_stub | grounding_iou | 0 | n/a | n/a | n/a | n/a | n/a |
| span_only | grounding_iou | 3 | 0.9697 | 0.005 | 0.007 | 0.9651 | 0.975 |
| span_verify | grounding_iou | 3 | 0.9928 | 0.0015 | 0.0022 | 0.9911 | 0.9938 |
| span_verify_norm | grounding_iou | 3 | 0.9928 | 0.0015 | 0.0022 | 0.9911 | 0.9938 |
| generative | strict_accuracy | 3 | 0.0477 | 0.0046 | 0.0065 | 0.0425 | 0.0512 |
| generative_verify | strict_accuracy | 3 | 0.0923 | 0.0056 | 0.008 | 0.0884 | 0.0988 |
| heuristic | strict_accuracy | 3 | 0.7593 | 0.0028 | 0.004 | 0.7572 | 0.7625 |
| llm_stub | strict_accuracy | 3 | 0.7574 | 0.0012 | 0.0017 | 0.7566 | 0.7588 |
| span_only | strict_accuracy | 3 | 0.8818 | 0.0096 | 0.0135 | 0.8759 | 0.8928 |
| span_verify | strict_accuracy | 3 | 0.8887 | 0.009 | 0.0127 | 0.8825 | 0.8991 |
| span_verify_norm | strict_accuracy | 3 | 0.9541 | 0.0095 | 0.0134 | 0.9481 | 0.965 |
<!-- /table -->

Three seeds of the identical configuration, varying both the initialisation and
the generated population — which is the honest thing to vary, because reusing one
dataset across seeds would report initialisation noise only and understate the
run-to-run scale.

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

Against the generative reference approach, every arm's accuracy gap is far outside
noise — `span_verify` at 65.206623x on canonical accuracy, and the largest ratio in
the table 119.208295x. Those numbers are large mostly because `generative` is both
very low and very stable: its canonical-accuracy noise scale is 0.006515, and the
verdict uses the larger of the two arms' scales. **A ratio of 65 is not a more
confident claim than one of 10**; it is a claim about a comparison whose
denominator is small, and it should be read as "far outside noise" and nothing
more precise. The grounding rows come back `unknown`
because generation makes no span claim at all: there is nothing to compare.

The more informative comparison is against the rule baseline, which is the
stronger competitor:

<!-- table:verdicts_vs_heuristic -->
| arm | metric | value | reference_value | delta | noise_scale | ratio_to_noise | verdict |
|---|---|---|---|---|---|---|---|
| generative | strict_accuracy | 0.0477 | 0.7593 | -0.7116 | 0.0065 | 109.2156 | robust |
| generative_verify | strict_accuracy | 0.0923 | 0.7593 | -0.667 | 0.008 | 83.8014 | robust |
| llm_stub | strict_accuracy | 0.7574 | 0.7593 | -0.0019 | 0.004 | 0.4676 | inside noise |
| span_only | strict_accuracy | 0.8818 | 0.7593 | 0.1225 | 0.0135 | 9.0537 | robust |
| span_verify | strict_accuracy | 0.8887 | 0.7593 | 0.1295 | 0.0127 | 10.1755 | robust |
| span_verify_norm | strict_accuracy | 0.9541 | 0.7593 | 0.1948 | 0.0134 | 14.5236 | robust |
| generative | canonical_accuracy | 0.0477 | 0.8244 | -0.7767 | 0.0065 | 119.2083 | robust |
| generative_verify | canonical_accuracy | 0.0923 | 0.8244 | -0.7321 | 0.008 | 91.9813 | robust |
| llm_stub | canonical_accuracy | 0.7574 | 0.8244 | -0.067 | 0.0031 | 21.6509 | robust |
| span_only | canonical_accuracy | 0.9444 | 0.8244 | 0.12 | 0.0122 | 9.798 | robust |
| span_verify | canonical_accuracy | 0.9546 | 0.8244 | 0.1302 | 0.0139 | 9.3623 | robust |
| span_verify_norm | canonical_accuracy | 0.9546 | 0.8244 | 0.1302 | 0.0139 | 9.3623 | robust |
| generative | coverage | 0.8852 | 0.7576 | 0.1276 | 0.0133 | 9.6121 | robust |
| generative_verify | coverage | 0.0013 | 0.7576 | -0.7564 | 0.0049 | 153.4852 | robust |
| llm_stub | coverage | 0.75 | 0.7576 | -0.0076 | 0.0119 | 0.6381 | inside noise |
| span_only | coverage | 0.9216 | 0.7576 | 0.164 | 0.0069 | 23.6059 | robust |
| span_verify | coverage | 0.8772 | 0.7576 | 0.1196 | 0.0195 | 6.1307 | robust |
| span_verify_norm | coverage | 0.8772 | 0.7576 | 0.1196 | 0.0195 | 6.1307 | robust |
| generative | grounding_exact | n/a | 0.9671 | n/a | 0.0043 | n/a | unknown |
| generative_verify | grounding_exact | n/a | 0.9671 | n/a | 0.0043 | n/a | unknown |
| llm_stub | grounding_exact | n/a | 0.9671 | n/a | 0.0043 | n/a | unknown |
| span_only | grounding_exact | 0.9666 | 0.9671 | -5.29e-04 | 0.0062 | 0.0853 | inside noise |
| span_verify | grounding_exact | 0.9927 | 0.9671 | 0.0256 | 0.0043 | 5.9244 | robust |
| span_verify_norm | grounding_exact | 0.9927 | 0.9671 | 0.0256 | 0.0043 | 5.9244 | robust |
| generative | grounding_iou | n/a | 0.969 | n/a | 0.004 | n/a | unknown |
| generative_verify | grounding_iou | n/a | 0.969 | n/a | 0.004 | n/a | unknown |
| llm_stub | grounding_iou | n/a | 0.969 | n/a | 0.004 | n/a | unknown |
| span_only | grounding_iou | 0.9697 | 0.969 | 6.84e-04 | 0.007 | 0.0972 | inside noise |
| span_verify | grounding_iou | 0.9928 | 0.969 | 0.0238 | 0.004 | 5.9089 | robust |
| span_verify_norm | grounding_iou | 0.9928 | 0.969 | 0.0238 | 0.004 | 5.9089 | robust |
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
| source | population | n_seeds | n_documents | n_opportunities | n_ungrounded | rate |
|---|---|---|---|---|---|---|
| enumerated spans (no model) | all admissible spans, all fields | 8 | 200 | 1.06e+06 | 0 | 0 |
| untrained span head | 25 documents/seed, verification off | 8 | 200 | 1598 | 0 | 0 |
| untrained gen. head | 25 documents/seed, verification off | 8 | 200 | 1567 | 1567 | 1 |
| trained span head | committed test-split per-item CSVs, verification off | 3 | 1200 | 8847 | 0 | 0 |
| trained gen. head | committed test-split per-item CSVs, verification off | 3 | 1200 | 8498 | 8486 | 0.998588 |
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
| arm | subset | n | strict_accuracy | canonical_accuracy | coverage |
|---|---|---|---|---|---|
| generative | needs_normalisation | 211 | 0 | 0 | 1 |
| generative_verify | needs_normalisation | 211 | 0 | 0 | 0.0047 |
| heuristic | needs_normalisation | 211 | 0 | 0.9336 | 0.981 |
| llm_stub | needs_normalisation | 211 | 0.7299 | 0.7299 | 0.7962 |
| span_only | needs_normalisation | 211 | 0 | 0.9573 | 1 |
| span_verify | needs_normalisation | 211 | 0 | 0.9953 | 1 |
| span_verify_norm | needs_normalisation | 211 | 0.9953 | 0.9953 | 1 |
| generative | verbatim | 2675 | 7.48e-04 | 7.48e-04 | 0.9279 |
| generative_verify | verbatim | 2675 | 7.48e-04 | 7.48e-04 | 0.0011 |
| heuristic | verbatim | 2675 | 0.7948 | 0.7981 | 0.8258 |
| llm_stub | verbatim | 2675 | 0.7301 | 0.7301 | 0.8295 |
| span_only | verbatim | 2675 | 0.9551 | 0.9555 | 0.9925 |
| span_verify | verbatim | 2675 | 0.9417 | 0.9424 | 0.9514 |
| span_verify_norm | verbatim | 2675 | 0.9417 | 0.9424 | 0.9514 |
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
| arm | coverage | absent_abstain_rate | n_absent | present_emit_rate | n_present | top_abstain_reason | top_abstain_count | mean_verify_iters |
|---|---|---|---|---|---|---|---|---|
| generative | 0.8909 | 0.4968 | 314 | 0.9331 | 2886 | no_candidate | 349 | 0 |
| generative_verify | 0.0013 | 1 | 314 | 0.0014 | 2886 | checks_failed | 2847 | 0.8897 |
| heuristic | 0.755 | 1 | 314 | 0.8371 | 2886 | no_candidate | 739 | 0.0316 |
| llm_stub | 0.7459 | 1 | 314 | 0.8271 | 2886 | no_candidate | 813 | 0 |
| span_only | 0.9163 | 0.7898 | 314 | 0.9931 | 2886 | no_candidate | 268 | 0 |
| span_verify | 0.8641 | 0.9713 | 314 | 0.955 | 2886 | no_candidate | 268 | 0.0709 |
| span_verify_norm | 0.8641 | 0.9713 | 314 | 0.955 | 2886 | no_candidate | 268 | 0.0709 |
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
| arm | ece | ace | mce | brier | nll | aurc | error_auroc | mean_confidence | n_calibration |
|---|---|---|---|---|---|---|---|---|---|
| generative | 0.417 | 0.417 | 0.5886 | 0.192 | 0.5683 | 0.9983 | 0.8459 | 0.4177 | 2851 |
| generative_verify | 0.0668 | 0.4963 | 0.5666 | 0.2508 | 0.6947 | 0.2708 | 0.75 | 0.5668 | 4 |
| heuristic | 0.1404 | 0.1366 | 0.5428 | 0.0846 | 0.4285 | 0.0209 | 0.6704 | 0.8389 | 2416 |
| llm_stub | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0 |
| span_only | 0.024 | 0.0254 | 0.0999 | 0.0446 | 0.1551 | 0.0085 | 0.9054 | 0.9174 | 2932 |
| span_verify | 0.0693 | 0.0693 | 0.4338 | 0.0362 | 0.1262 | 0.0014 | 0.9058 | 0.9184 | 2765 |
| span_verify_norm | 0.0693 | 0.0693 | 0.4338 | 0.0362 | 0.1262 | 0.0014 | 0.9058 | 0.9184 | 2765 |
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
| variant | split | canonical_accuracy | strict_accuracy | coverage | n_records |
|---|---|---|---|---|---|
| row_only | validation | 0.8267 | 0.7575 | 0.7575 | 1200 |
| row_first | validation | 0.8267 | 0.7575 | 0.8083 | 1200 |
| below_cheap_last | validation | 0.8025 | 0.7333 | 0.7883 | 1200 |
| below_cheap | validation | 0.7967 | 0.7217 | 0.7833 | 1200 |
| row_first_wide | validation | 0.7742 | 0.705 | 0.775 | 1200 |
| any_page | validation | 0.7742 | 0.705 | 0.775 | 1200 |
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
| kind | ablation | switch | metric | value | full_value | delta | noise_scale | ratio_to_noise | verdict | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| inference | conf_threshold_0.5 | min_confidence=0.5 | canonical_accuracy | 0.9256 | 0.9546 | -0.029 | 0.0139 | 2.0822 | survives | 3 |
| inference | no_abstain | abstain=False | canonical_accuracy | 0.9589 | 0.9546 | 0.0043 | 0.0139 | 0.3071 | inside noise | 3 |
| inference | no_arithmetic | arithmetic=False | canonical_accuracy | 0.9624 | 0.9546 | 0.0078 | 0.0139 | 0.5617 | inside noise | 3 |
| inference | no_type_check | type_check=False | canonical_accuracy | 0.9376 | 0.9546 | -0.017 | 0.0139 | 1.2208 | suggestive | 3 |
| inference | no_verification | enabled=False | canonical_accuracy | 0.9444 | 0.9546 | -0.0102 | 0.0139 | 0.734 | inside noise | 3 |
| inference | conf_threshold_0.5 | min_confidence=0.5 | coverage | 0.85 | 0.8772 | -0.0272 | 0.0195 | 1.3938 | suggestive | 3 |
| inference | no_abstain | abstain=False | coverage | 0.9216 | 0.8772 | 0.0444 | 0.0195 | 2.275 | survives | 3 |
| inference | no_arithmetic | arithmetic=False | coverage | 0.9178 | 0.8772 | 0.0406 | 0.0195 | 2.0827 | survives | 3 |
| inference | no_type_check | type_check=False | coverage | 0.8919 | 0.8772 | 0.0147 | 0.0195 | 0.753 | inside noise | 3 |
| inference | no_verification | enabled=False | coverage | 0.9216 | 0.8772 | 0.0444 | 0.0195 | 2.275 | survives | 3 |
| inference | conf_threshold_0.5 | min_confidence=0.5 | grounding_exact | 0.9946 | 0.9927 | 0.0019 | 0.0022 | 0.8477 | inside noise | 3 |
| inference | no_abstain | abstain=False | grounding_exact | 0.9827 | 0.9927 | -0.0101 | 0.0022 | 4.6031 | robust | 3 |
| inference | no_arithmetic | arithmetic=False | grounding_exact | 0.984 | 0.9927 | -0.0087 | 0.0022 | 3.9721 | robust | 3 |
| inference | no_type_check | type_check=False | grounding_exact | 0.9713 | 0.9927 | -0.0214 | 0.0022 | 9.8185 | robust | 3 |
| inference | no_verification | enabled=False | grounding_exact | 0.9666 | 0.9927 | -0.0261 | 0.0022 | 11.9492 | robust | 3 |
| inference | conf_threshold_0.5 | min_confidence=0.5 | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| inference | no_abstain | abstain=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| inference | no_arithmetic | arithmetic=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| inference | no_type_check | type_check=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| inference | no_verification | enabled=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| inference | conf_threshold_0.5 | min_confidence=0.5 | strict_accuracy | 0.8639 | 0.8887 | -0.0249 | 0.0127 | 1.9565 | suggestive | 3 |
| inference | no_abstain | abstain=False | strict_accuracy | 0.8857 | 0.8887 | -0.003 | 0.0127 | 0.2374 | inside noise | 3 |
| inference | no_arithmetic | arithmetic=False | strict_accuracy | 0.8969 | 0.8887 | 0.0081 | 0.0127 | 0.6385 | inside noise | 3 |
| inference | no_type_check | type_check=False | strict_accuracy | 0.8747 | 0.8887 | -0.0141 | 0.0127 | 1.1051 | suggestive | 3 |
| inference | no_verification | enabled=False | strict_accuracy | 0.8818 | 0.8887 | -0.007 | 0.0127 | 0.5485 | inside noise | 3 |
| training | no_2d_pos | use_2d_pos=False | canonical_accuracy | 0.9293 | 0.9546 | -0.0253 | 0.0139 | 1.82 | suggestive | 3 |
| training | no_spatial_bias | use_spatial_bias=False | canonical_accuracy | 0.9096 | 0.9546 | -0.045 | 0.0139 | 3.2356 | robust | 3 |
| training | no_2d_pos | use_2d_pos=False | coverage | 0.8782 | 0.8772 | 0.001 | 0.0195 | 0.0534 | inside noise | 3 |
| training | no_spatial_bias | use_spatial_bias=False | coverage | 0.84 | 0.8772 | -0.0372 | 0.0195 | 1.9065 | suggestive | 3 |
| training | no_2d_pos | use_2d_pos=False | grounding_exact | 0.9724 | 0.9927 | -0.0203 | 0.0022 | 9.3033 | robust | 3 |
| training | no_spatial_bias | use_spatial_bias=False | grounding_exact | 0.9867 | 0.9927 | -0.006 | 0.0022 | 2.7317 | survives | 3 |
| training | no_2d_pos | use_2d_pos=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| training | no_spatial_bias | use_spatial_bias=False | hallucination_rate | 0 | 0 | 0 | 0 | n/a | unknown | 3 |
| training | no_2d_pos | use_2d_pos=False | strict_accuracy | 0.8674 | 0.8887 | -0.0214 | 0.0127 | 1.6782 | suggestive | 3 |
| training | no_spatial_bias | use_spatial_bias=False | strict_accuracy | 0.845 | 0.8887 | -0.0437 | 0.0127 | 3.4382 | robust | 3 |
<!-- /table -->

Two families, and keeping them apart matters. **Inference-time** switches reuse
the *same trained weights* and the same decoded candidate sets, so their deltas
contain no initialisation noise at all — the noise scale they are divided by is
the `full` arm's seed-to-seed spread, which asks whether the switch's effect is
larger than the variability of the pipeline it sits inside. **Training-time**
switches retrain, so their deltas carry run-to-run noise directly.

**Read the grounding column first.** Every switch shows up there, robustly, and
almost nothing shows up in accuracy:

| switch | grounding delta | ratio | verdict | strict-accuracy ratio |
|---|---|---|---|---|
| verification off | -0.026098 | 11.949206 | robust | 0.548481 (inside noise) |
| type check off | -0.021445 | 9.818455 | robust | 1.105148 (suggestive) |
| abstention off | -0.010054 | 4.603053 | robust | 0.237402 (inside noise) |
| arithmetic check off | -0.008675 | 3.972056 | robust | 0.638530 (inside noise) |
| 2-D position off | -0.020319 | 9.303289 | robust | 1.678187 (suggestive) |
| spatial bias off | -0.005966 | 2.731737 | survives | 3.438237 (robust) |

That is the honest shape of this result. **The mechanisms this repository adds
are grounding mechanisms.** They move the extractor onto the right span, reliably
and well outside noise. What they do to end-to-end accuracy is mostly not
measurable at this scale, and in one case points the wrong way: turning the
arithmetic check *off* raises canonical accuracy by 0.007812 (0.561739x, inside
noise) while costing 0.008675 grounding exactness. The check refuses answers that
were often right.

Three switches are worth calling out individually.

**The spatial attention bias is the one mechanism that buys accuracy.** Turning it
off costs 0.043750 strict accuracy at 3.438237x noise and 0.045000 canonical at
3.235615x -- both robust -- from a table of 332 parameters. The absolute 2-D
position encoding, which is far larger, costs only 0.021354 strict at 1.678187x,
merely suggestive. The *relational* signal is what matters, which is what
`docs/METHOD.md` section 3.2 argued it would be.

**The confidence threshold is a bad idea and the table says so.** Setting
`verify.min_confidence=0.5` costs 0.028958 canonical accuracy at 2.082178x
(survives) and 0.027188 coverage, and buys 0.001852 grounding at 0.847745x --
inside noise. It is shipped as a switch and left at 0.0.

**Abstention is worth more than it costs.** Turning it off raises coverage by
0.044375 (2.274987x, survives) and changes strict accuracy by -0.003021
(0.237402x, inside noise) while costing 0.010054 grounding exactness at 4.603053x.
Emitting a value that failed its checks does not make the arm more accurate; it
makes it less grounded.

**Hallucination rate is 0.000000 under every switch**, which is why those rows
read `unknown`: with no variance across seeds there is no noise scale to divide
by. That is the correct output for a quantity that is constant by construction,
and it is worth seeing: no inference-time switch -- not even turning verification
off entirely -- can make a selection head emit a string absent from the document.

### 7.1 The verification loop, isolated

`span_only` is the same weights and the same decoded candidates with the loop
switched off, so comparing against it attributes the loop exactly:

<!-- table:verdicts_vs_span_only -->
| arm | metric | value | reference_value | delta | noise_scale | ratio_to_noise | verdict |
|---|---|---|---|---|---|---|---|
| generative | strict_accuracy | 0.0477 | 0.8818 | -0.8341 | 0.0135 | 61.6434 | robust |
| generative_verify | strict_accuracy | 0.0923 | 0.8818 | -0.7895 | 0.0135 | 58.3484 | robust |
| heuristic | strict_accuracy | 0.7593 | 0.8818 | -0.1225 | 0.0135 | 9.0537 | robust |
| llm_stub | strict_accuracy | 0.7574 | 0.8818 | -0.1244 | 0.0135 | 9.1922 | robust |
| span_verify | strict_accuracy | 0.8887 | 0.8818 | 0.007 | 0.0135 | 0.5158 | inside noise |
| span_verify_norm | strict_accuracy | 0.9541 | 0.8818 | 0.0723 | 0.0135 | 5.3429 | robust |
| generative | canonical_accuracy | 0.0477 | 0.9444 | -0.8967 | 0.0122 | 73.2125 | robust |
| generative_verify | canonical_accuracy | 0.0923 | 0.9444 | -0.8521 | 0.0122 | 69.5723 | robust |
| heuristic | canonical_accuracy | 0.8244 | 0.9444 | -0.12 | 0.0122 | 9.798 | robust |
| llm_stub | canonical_accuracy | 0.7574 | 0.9444 | -0.187 | 0.0122 | 15.2668 | robust |
| span_verify | canonical_accuracy | 0.9546 | 0.9444 | 0.0102 | 0.0139 | 0.734 | inside noise |
| span_verify_norm | canonical_accuracy | 0.9546 | 0.9444 | 0.0102 | 0.0139 | 0.734 | inside noise |
| generative | coverage | 0.8852 | 0.9216 | -0.0364 | 0.0133 | 2.7385 | survives |
| generative_verify | coverage | 0.0013 | 0.9216 | -0.9203 | 0.0069 | 132.5018 | robust |
| heuristic | coverage | 0.7576 | 0.9216 | -0.164 | 0.0069 | 23.6059 | robust |
| llm_stub | coverage | 0.75 | 0.9216 | -0.1716 | 0.0119 | 14.3976 | robust |
| span_verify | coverage | 0.8772 | 0.9216 | -0.0444 | 0.0195 | 2.275 | survives |
| span_verify_norm | coverage | 0.8772 | 0.9216 | -0.0444 | 0.0195 | 2.275 | survives |
| generative | grounding_exact | n/a | 0.9666 | n/a | 0.0062 | n/a | unknown |
| generative_verify | grounding_exact | n/a | 0.9666 | n/a | 0.0062 | n/a | unknown |
| heuristic | grounding_exact | 0.9671 | 0.9666 | 5.29e-04 | 0.0062 | 0.0853 | inside noise |
| llm_stub | grounding_exact | n/a | 0.9666 | n/a | 0.0062 | n/a | unknown |
| span_verify | grounding_exact | 0.9927 | 0.9666 | 0.0261 | 0.0062 | 4.2071 | robust |
| span_verify_norm | grounding_exact | 0.9927 | 0.9666 | 0.0261 | 0.0062 | 4.2071 | robust |
| generative | grounding_iou | n/a | 0.9697 | n/a | 0.007 | n/a | unknown |
| generative_verify | grounding_iou | n/a | 0.9697 | n/a | 0.007 | n/a | unknown |
| heuristic | grounding_iou | 0.969 | 0.9697 | -6.84e-04 | 0.007 | 0.0972 | inside noise |
| llm_stub | grounding_iou | n/a | 0.9697 | n/a | 0.007 | n/a | unknown |
| span_verify | grounding_iou | 0.9928 | 0.9697 | 0.0231 | 0.007 | 3.2874 | robust |
| span_verify_norm | grounding_iou | 0.9928 | 0.9697 | 0.0231 | 0.007 | 3.2874 | robust |
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

## 8. Determinism

<!-- table:determinism -->
| quantity | n | max_abs_difference | identical |
|---|---|---|---|
| generator_token_boxes | 13812 | 0 | yes |
| generator_token_texts | 3453 | 0 | yes |
| span_verify_confidence | 3200 | 0 | yes |
| span_verify_emitted_values | 3200 | 0 | yes |
<!-- /table -->

Two separate invocations, per-item outputs diffed: 13812 generated box
coordinates, 3453 token strings, and 3200 per-item confidences and emitted
strings from `span_verify`. Every maximum absolute difference is exactly 0.0 and
every comparison is identical. The generator is a pure function of
`(doc_id, seed, config)` and the evaluation of a fixed checkpoint is
deterministic. String-valued quantities
report `0.0` when identical and `n/a` otherwise, because there is no metric on
strings and a fabricated one would be worse than an honest gap.

Two design choices make this hold, and both are the kind of thing that fails
silently if forgotten. The hash-bucketed vocabulary uses `zlib.crc32` rather than
Python's `hash`, which is salted per process and would otherwise make features
depend on which process built them. And splits are disjoint document-id ranges
rather than a shuffled pool, so nothing can leak between them because nothing is
shared.

## 9. Retractions and negative results

### 9.1 Retracted: "selection prevents hallucination", as a claim about heads

The claim this repository set out to make was that a selection head prevents
hallucination where a generative head does not. The narrower claim is what the
evidence supports: **the provenance check prevents hallucination, from either
head.** `generative_verify` — the reference approach with the identical
verification loop bolted on — reaches a hallucination rate of exactly 0.0.

What separates the two is the *price*. `generative_verify` pays 0.001250 coverage
against `span_verify`'s 0.877188, emitting 4 values on seed 0 where the span arm
emits 2765, and it burns 7.1175 verification iterations per document doing it.
Selection satisfies the check for free and needs no checker at all to be safe;
that is a real and useful difference, and it is a smaller claim than the one this
project started with.

This was only visible because the arm capable of refuting the broad claim was
built and reported. It would have been easy not to build it.

### 9.2 Retracted: an earlier version of the guarantee was false

Before the admissibility mask confined spans to a single page, a span could
concatenate the end of one page with the start of the next. Its rendering is a
sequence of document tokens but **not** a contiguous region: it has no union box,
and the provenance predicate correctly refuses to find it.
`test_untrained_span_model_never_hallucinates` failed on 3 of 8 seeds with an
emitted `po_number` of `"of 2 Invoice No"` spliced across a page break.

The general lesson is stated in `docs/METHOD.md` §2.1 and is worth repeating: **a
guarantee about an output space is only as good as the agreement between the
decoder's mask and the checker's definition.** The guarantee was not wrong in
principle; the implementation of the output space was, and only an exhaustive test
found it.

### 9.3 Weakened: the accuracy gap over the generative baseline

`generative` reaches 0.047708 canonical accuracy (3-seed mean, sd 0.004607). That
is a real equal-budget result — same encoder, same data, same seed, same schedule,
which is exactly the comparison the protocol prescribes — but it is *not* evidence
that generation cannot do this task.

<!-- table:generative_budget -->
| arm | epochs | status | strict_accuracy | canonical_accuracy | coverage | hallucination_rate | final_val_loss | n_records |
|---|---|---|---|---|---|---|---|---|
| generative (shared budget) | 6 | evaluated | 0.0494 | 0.0494 | 0.8909 | 0.9986 | 1.0016 | 3200 |
| generative (long schedule) | 7 | training curve only, not evaluated | n/a | n/a | n/a | n/a | 0.9645 | n/a |
<!-- /table -->

A character decoder has to learn to spell before it can be right: at roughly 1.0
nats per character, per-character accuracy of around two thirds compounds to a few
percent over a ten-character value, which is what is observed.

**A longer schedule was attempted twice and both runs were killed by this
machine's background-job limit before their evaluation stage.** What survives is
the training curve, and it is committed
(`results/runs/gdx_generative_long/history.jsonl`): validation loss reaches
0.964541 by epoch 7 against 1.001558 at the shared budget's
epoch 6. So longer training does keep improving the decoder, slowly, and the
accuracy that would result **was not measured**. The table above says
`training curve only, not evaluated` for that row and leaves its accuracy columns
`n/a`, because inferring them would be inventing a number.

Two consequences, stated so no reader has to work them out:

* The equal-budget comparison is the one the protocol prescribes and it stands.
  The *ranking* is not in doubt: nothing about a validation loss falling from
  1.001558 to 0.964541 closes a gap of 0.906875 canonical
  accuracy.
* **The magnitude of the gap is not a stable estimate** and should not be quoted
  as one. Anyone wanting the converged number has to run the schedule to
  completion; the command is in `docs/REPRODUCIBILITY.md`.

### 9.4 Negative: the verification loop does not improve accuracy

+0.010208 canonical accuracy at 0.734005x the run-to-run noise scale, and
+0.006979 strict at 0.515812x. Both **inside noise**. The loop demonstrably
improves grounding (+0.026098 at 4.207149x) and correct abstention (0.789809 to
0.971338) and demonstrably costs coverage (−0.044375 at 2.274987x), but its net
effect on accuracy over all pairs is not distinguishable from reseeding.

The headline framing "verification makes the method more accurate" is therefore
not supported and is not used.

### 9.5 Negative: zero accuracy where normalisation is required

0.000000 strict accuracy on the 211 test fields whose written form is not the
target — for this method, for its verification-free ablation, and for the rule
baseline alike. This is a structural bound on selection, not a training failure,
and no amount of scale changes it. `span_verify_norm` recovers it to 0.995261 by
adding a deterministic parser, at the cost of the substring guarantee.

### 9.6 Negative: verification damages calibration

`span_verify` is more accurate than `span_only` and *worse* calibrated: ECE
0.069292 against 0.024006, MCE 0.433785 against 0.099913. Filtering by a check
removes low-confidence errors and leaves the surviving confidences systematically
too low relative to a now-higher accuracy. Both directions are reported.

### 9.7 Negative: the learned head's grounding advantage is inside noise

`span_only` versus `heuristic` on exact-span grounding: −0.000529, a ratio of
0.085287 — **inside noise**. Without the verification loop, a 108892-parameter
layout transformer is not better at *looking in the right place* than a geometric
rule with a synonym table. Every grounding advantage this repository reports comes
from the loop, not from the head.

### 9.8 Design errors found and fixed during the build

Each of these produced a plausible wrong number before it was caught, and each has
a regression test.

* **A permissive amount parser** stripped every non-digit character, so
  `"Net Amount EUR 30,614.90"` parsed as an amount. That turned label text into a
  value and inflated the count of document spans that "contain" a value — which
  weakens the hallucination metric *in this method's favour*.
* **A permissive date parser** ignored unrecognised words, so
  `"Issued 29th of March 2024"` parsed to the same date as
  `"29th of March 2024"` and the label token counted as part of a correct
  grounding.
* **A label matcher that stopped at the first mismatch** meant only the longest
  synonym was ever tried; the rule baseline found labels on about 20% of fields
  instead of 96%, and reporting that would have been a strawman.
* **Colliding magnitude buckets** made `0.05` and `5.00` the same input feature.
* **A pure-Python decode loop** was about 30x slower than the vectorised version.
  That is a measurement bug, not just a slow one: it would have made the selection
  head look slower than the generative head and inverted the efficiency claim.
* **Page-crossing spans**, as above.

## 10. Limitations, and what would change them

**The data is synthetic, and that is a load-bearing limitation.** It is also the
reason the grounding oracle exists at all: no annotated real dataset records the
token span each value was read from, so on FUNSD, CORD or SROIE "did the model
look in the right place" is only answerable by string matching, which scores a
model that reads the right value from the wrong place as correct. The trade is
deliberate and it cuts both ways. What would change it: a real dataset with
span-level provenance annotation, or a human study on a sample of real invoices.

**Reading order here is the order the generator wrote in.** Real OCR order is
noisier, and the 2-D positional encodings exist partly to survive that — but this
benchmark never tests it, because its token order is always correct. A shuffled
reading-order condition would be a cheap and informative addition and is not here.

**No recognition errors.** Real scans produce `l` for `1` and drop characters.
Every token in this benchmark is spelled correctly, which flatters *every* arm and
flatters selection most: a selection head cannot repair a misread token, while a
generative head in principle can. This is the most likely place where the ranking
reported here would change on real data.

**One architecture, one scale, three seeds.** A 64-wide 2-layer encoder trained
for 6 epochs on 1200 documents. Nothing here says how the comparison behaves at a
scale where the generative head can spell.

**Greedy decoding for both heads, no beam.** A beam would improve the generative
baseline's strings and would give it the ranked candidate set the verification
loop needs to *re-select* rather than only abstain. That is the most obvious way
to strengthen the baseline and it was not done, for compute reasons.

**The LLM arm is a deterministic offline stub.** Its accuracy numbers are the
stub's, not any language model's, and nothing in this file claims otherwise. Its
token counts are whitespace-based and therefore a lower bound. What would change
it: running the same `LLMClient` against a real provider, which the code supports
and this study did not do.

**The `n=4` calibration row.** `generative_verify`'s calibration statistics are
computed over four emitted values on seed 0. They are kept for completeness and
are meaningless; they should not be quoted.

**Statistical unit.** The paired tests condition on one trained model per arm and
are statements about weights. The seed study is the method-level claim, and with
three seeds its noise scale is itself estimated from two degrees of freedom. Every
verdict in this file should be read with that in mind.
