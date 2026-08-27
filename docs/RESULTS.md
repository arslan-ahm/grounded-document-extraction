# Results

Every number in this file is injected from a CSV in `results/tables/` by
`scripts/render_docs.py`. None of them is typed by hand, and
`tests/test_report_and_cli.py::test_shipped_documents_are_not_stale` fails if any
has drifted from its source.

`PENDING_NUMBERS`

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
2-layer encoder, 1200 training documents, 6 epochs, three seeds, ~95 minutes of
total compute on two CPU threads. The structural claims are scale-free. The
accuracy comparisons are the ones a larger study could move.

## 1. Efficiency

<!-- table:efficiency -->
<!-- /table -->

<!-- table:decode_scaling -->
<!-- /table -->

<!-- table:baseline_cost -->
<!-- /table -->

`PENDING_NUMBERS`

## 2. The seven-arm comparison

<!-- table:method -->
<!-- /table -->

`PENDING_NUMBERS`

### 2.1 Per field

<!-- table:per_field -->
<!-- /table -->

`PENDING_NUMBERS`

### 2.2 Is the difference significant?

<!-- table:statistical_tests -->
<!-- /table -->

`PENDING_NUMBERS`

### 2.3 And is it bigger than the noise?

<!-- table:seed_variance -->
<!-- /table -->

<!-- table:verdicts -->
<!-- /table -->

`PENDING_NUMBERS`

## 3. The hallucination result, stated precisely

`PENDING_NUMBERS`

## 4. The measured cost of the ideology

<!-- table:normalisation_cost -->
<!-- /table -->

`PENDING_NUMBERS`

## 5. Abstention and calibration

<!-- table:abstention -->
<!-- /table -->

<!-- table:calibration -->
<!-- /table -->

`PENDING_NUMBERS`

## 6. Ablations

<!-- table:ablations -->
<!-- /table -->

`PENDING_NUMBERS`

## 7. The rule baseline, given a fair fight

<!-- table:heuristic_sweep -->
<!-- /table -->

`PENDING_NUMBERS`

## 8. Determinism

<!-- table:determinism -->
<!-- /table -->

`PENDING_NUMBERS`

## 9. Retractions and negative results

`PENDING_NUMBERS`

## 10. Limitations, and what would change them

`PENDING_NUMBERS`
