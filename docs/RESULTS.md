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
