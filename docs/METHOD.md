# Method

## 1. The structural assumption being replaced

The reference approach to extracting fields from a visually-rich document is: run
a model over the page, have it emit a **string** per field, and trust the string.
Concretely, for a document `D` and a field `f`, the model defines a distribution
over sequences and the output is

```
ŷ_f  =  argmax_{y ∈ Σ*}  p(y | D, f)
```

where `Σ*` is the free monoid over some alphabet. The output space is *all
strings*. Nothing in that formulation refers to `D` except through the
conditioning, so the emitted string carries no pointer back into the document.
Two consequences follow, and they are the reason for this repository:

1. **A hallucinated value and a correctly-read value are the same kind of
   object.** Both are elements of `Σ*`. There is no query you can put to the
   output that distinguishes them.
2. **Nothing downstream is checkable.** If `subtotal`, `tax` and `total` come back
   as three strings, they can be mutually inconsistent, and there is no way to ask
   which of the three was misread, because none of them is attached to anything.

## 2. The replacement: selection plus verification

Let `D = (t_1, b_1), …, (t_n, b_n)` be the document's tokens, each a string `t_i`
and a box `b_i ∈ [0,1]^4` on page `π_i`. Define the **admissible span set**

```
A(D)  =  { (s, e) : 1 ≤ s ≤ e ≤ n,  e - s + 1 ≤ L,  π_s = π_e }  ∪  { ⊥ }
```

where `L` is `model.max_span_len` and `⊥` is abstention. The extractor is a map

```
g_f : D  ↦  A(D)
```

and the emitted value is the *rendering* of the selected span,
`val(s,e) = t_s ⊕ " " ⊕ … ⊕ " " ⊕ t_e`, or nothing when `⊥` is chosen.

Two properties are now theorems rather than measurements:

**(P1) The emitted string is a substring of the document.** Immediate:
`val(s,e)` is a concatenation of consecutive tokens of `D`. So for any string `y`
that does not occur as a contiguous token span of `D`, `P(ŷ_f = y) = 0` exactly.
Not small — zero, for every parameter setting, trained or not.

**(P2) The emitted value has a location.** `(s,e)` is the provenance, and
`box(s,e) = ⋃_{i=s..e} b_i` is a region of one page. This is what makes grounding
accuracy measurable and what makes cross-field checks possible.

### 2.1 Why the page-confinement clause is in `A(D)`

`π_s = π_e` looks like a detail. It is load-bearing. Drop it and a span can
concatenate the last tokens of page 1 with the first tokens of page 2 — a
sequence of document tokens whose rendering is **not** a contiguous region, has no
union box, and does not occur as a readable substring of the document. (P1) then
fails against any provenance predicate that respects page boundaries, which every
sensible one does.

This was not a hypothetical. `test_untrained_span_model_never_hallucinates`
failed on 3 of 8 seeds before the clause was added, with an emitted `po_number` of
`"of 2 Invoice No"` spliced across a page break. The general lesson is worth
stating: **(P1) holds only when the candidate space is exactly the set of spans
the provenance predicate accepts.** A guarantee about an output space is only as
good as the agreement between the decoder's mask and the checker's definition.

### 2.2 The bounded verification loop

Selection makes checks possible; the loop is where they are spent. Three
predicates, applied to a ranked candidate list per field:

* **Provenance.** `val(c) ∈ spans(D)` under a canonicalising comparison
  (§4). Trivially true for a span candidate; a real test for a generated string.
* **Type.** `val(c)` could be an instance of `f`'s type — an amount parses as a
  number, a date parses to an ISO date, an id matches its pattern. A *necessary*
  condition only: it cannot tell a subtotal from a total.
* **Arithmetic.** `|num(subtotal) + num(tax) − num(total)| ≤ τ`, evaluated on
  three separately-grounded spans. `τ = 0.011` currency units; the generator does
  its arithmetic in integer cents so the relation is exact in the data and any
  violation at inference is a real inconsistency rather than a rounding artefact.

The loop is:

```
for each field f:                                   # local checks
    for k = 1 … min(K, |C_f|):
        if type(c_k) and provenance(c_k): emit c_k; break
    else: abstain(f)

for i = 1 … K:                                      # global repair
    if arithmetic(subtotal, tax, total) is not False: break
    advance the least-confident participating field to its
        next locally-valid candidate; if none exists, stop
if arithmetic(…) is False: abstain on all three amount fields
```

`K = verify.max_iters = 3`. The bound matters: an unbounded repair loop is not an
algorithm with a runtime. The worst case here is three candidate advances, and
the realised iteration count is recorded per document so the cost appears in the
results rather than in a footnote.

**When the arithmetic check does not apply it returns `None`, not `True`.** The
generator drops `subtotal` or `tax` on a fraction of documents; on those the
constraint has no content. Returning `True` there would inflate the reported
consistency rate by counting documents the check never ran on.

### 2.3 What abstention means

`⊥` is a first-class output, trained by the same cross-entropy as every other
position: the sequence carries a `[CLS]` token at index 0 and an absent field's
start and end targets both point there. So abstention needs no separate
classifier and no separately-tuned threshold.

Abstention is scored as **correct exactly when the field is genuinely absent**. A
method cannot buy accuracy by declining to answer, and `absent_abstain_rate` is
reported next to `coverage` so a high-accuracy, low-coverage arm cannot hide.

## 3. The model

### 3.1 Inputs

Per token: a word identity, a surface shape class, a numeric magnitude bucket, and
ten geometry scalars.

**Word identity** is a closed lexicon plus a fixed hash tail. Numeric literals are
unbounded, so anything outside the lexicon is hashed. The hash is `zlib.crc32`,
not Python's `hash`, because `hash` on `str` is salted per process — using it
would make the features depend on which process built them, which is a
reproducibility bug that does not announce itself.

**Shape class** (17 values) separates `INV-81344`, `2024-01-03`, `$1,234.56` and
`Bracket` regardless of identity. With a 308-bucket hash tail the model cannot
recover that from identity alone, and it is the feature that lets the model and
the type check agree about what a token *could* be.

**Magnitude bucket** (9 values) is the log-scale magnitude of a numeric token.
Bucket 1 is sub-unit and buckets `k ≥ 2` hold `[10^(k-2), 10^(k-1))`. The
arithmetic relation between subtotal, tax and total is a relation between
magnitudes; giving the model the bucket makes it representable without
reconstructing place value from a hashed string. (An earlier version collapsed
sub-unit values into the `[1,10)` bucket, so `0.05` and `5.00` were the same
feature; `test_magnitude_bucket_separates_sub_unit_from_single_digit` pins the
fix.)

### 3.2 Positional encodings — three signals, kept separate

**Reading order** — fixed sinusoids over the token index, always on. This matters
for the ablation: if `use_2d_pos=false` removed *all* position information the
comparison would be "layout-aware versus bag of tokens" and any gap would be
uninterpretable. The honest contrast is **2-D layout position against 1-D
reading-order position**, and that is what the switch does.

**2-D layout position** — Fourier features over five geometric scalars (box
centre `c_x, c_y`, box size `w, h`, page fraction), projected to the model width:

```
γ(v)  =  [ sin(2^0 π v), cos(2^0 π v), …, sin(2^{B-1} π v), cos(2^{B-1} π v) ]
```

with `B = 8` bands. This is the standard random-Fourier / positional-encoding
trick (Tancik et al., 2020; Mildenhall et al., 2020). A linear layer on raw
coordinates cannot represent "these two tokens are on the same row" without an
enormous Lipschitz constant, whereas `sin(2^k π y)` makes near-equality of `y` a
low-frequency agreement. At 8 bands the finest band resolves about 0.8% of the
page height, which is finer than one text line.

**Spatial attention bias** — a *relative* signal, and the part absolute encodings
cannot supply. "The value is to the right of its label" and "the value is on the
row below its label" are relations, not positions. The bias added to the attention
logits before the softmax is

```
β_h(i, j)  =  T_h[ bx(c_x^j − c_x^i), by(c_y^j − c_y^i) ]  +  P_h[ 𝟙(π_i = π_j) ]
```

where `bx, by` are signed log-scale bucketings into 9 buckets each. This is the
T5 relative-position mechanism (Raffel et al., 2020) generalised from one
dimension to two. The bucketing is log-scale because layout relations live near
zero: a label and its value are a few percent of the page apart, while the
difference between 30% and 60% carries almost no information. The whole table
costs `4 × 9 × 9 + 4 × 2 = 332` parameters, which is the point — the prior is
nearly free, and the ablation measures whether it is worth anything.

### 3.3 Encoder

Two pre-norm transformer blocks, `d_model = 64`, 4 heads, `d_ff = 128`.
Hand-rolled attention rather than `nn.MultiheadAttention` for two reasons: the
spatial bias has to reach the attention *logits*, and the mechanism that makes
the encoder layout-aware should be fifteen readable lines that a test can check
against a hand-computed reference.

Pre-norm rather than post-norm because this encoder trains for a few hundred
steps and has no warm-up budget to spare; post-norm needs a longer warm-up to
survive the early-training gradient spike, and those steps would come out of the
project's total compute budget.

No pretrained checkpoint, no `transformers`, no `detectron2`, no
`torch-geometric`. That is a deliberate constraint: the claim under test is about
the *output interface*, and the cleanest way to isolate it is to hold one small,
fully-visible backbone fixed and change only the head. A pretrained encoder would
make the comparison depend on what the checkpoint had already memorised about
invoices.

### 3.4 The two heads

**Span head.** One `Linear(d_model, 2F)`, giving per-field start and end logits
over positions. The factorisation `p(s,e) = p_start(s) · p_end(e)` is the standard
extractive-QA one (Devlin et al., 2019). It is not exact — start and end are not
independent — but combined with the admissibility mask and renormalisation over
the decision set

```
Z  =  p_start(0)p_end(0)  +  Σ_{(s,e) ∈ A(D)} p_start(s)p_end(e)
```

it yields a proper distribution over the options the head can actually choose,
which is all the calibration metrics need. An unnormalised span score is not a
probability and calibrating it would be meaningless.

**Generative head — the reference approach.** A character-level attentional GRU
decoder over a closed 70-character alphabet, one shared decoder conditioned on a
field embedding, with additive attention over the encoder states (Bahdanau et al.,
2015). Two fairness decisions:

* It attends over the encoder states rather than over a pooled vector. A
  pooled-only decoder could not read a specific number off the page at all, and
  beating that would prove nothing.
* One shared decoder, not eight. Eight decoders would give the baseline capacity
  the span head does not have.

It abstains by emitting the empty string, which the training targets teach
directly (an absent field's target is `""`). Both heads therefore have the same
abstention affordance.

**The asymmetry that is real and is reported.** A span head supplies a ranked
candidate set for free — the top-`k` of a distribution over `A(D)`. A generative
head does not: its "second-best string" requires a beam. So the verification loop
can *re-select* for the span arms but can only *accept or abstain* for the
generative arms. This is a genuine advantage of selection, not a modelling
choice, and `docs/RESULTS.md` states it where the arms are compared.

### 3.5 One training loop

`model.head` is the only configuration difference between the method and its
reference baseline. Everything else — width, depth, positional encodings,
optimiser, schedule, data, seed — is held fixed *by construction*, because
separate scripts per arm are how an incidental difference in schedule gets
reported as a difference in method.

Checkpoint selection is on **validation loss**, not on a validation metric. The
two heads' natural metrics are not comparable (span exactness versus character
accuracy), so selecting on a metric would apply a different rule to each arm.
Validation loss is each head's own objective on held-out data and is the only
criterion that is the same *rule* for both.

Both heads are built after re-seeding on the run's seed, so a given seed gives
both arms the same encoder initialisation. Without that, part of the
span-versus-generation gap would be an initialisation difference.

## 4. Comparison semantics

Three notions of correctness, reported side by side because collapsing them would
hide this project's central limitation.

`strict` — equal after whitespace and case normalisation only. This is what a
consumer that needs ISO dates actually requires.

`canonical` — equal after parsing dates to ISO and amounts to numbers. The
generous reading.

`ANLS` — thresholded normalised edit similarity (Biten et al., 2019), so a
near-miss earns partial credit only above 0.5 and otherwise scores zero.

**The gap between `strict` and `canonical`, restricted to fields whose written
form differs from the target, is the measured cost of selection-only
extraction.** With probability `data.verbose_date_prob = 0.30` a date is written
in a non-ISO form — `03/01/2024`, `Jan 3, 2024`, `3rd of Jan 2024`. No contiguous
span equals `2024-01-03`, so a selection-only extractor's strict accuracy there is
**bounded at zero by construction**. `results/tables/normalisation_cost.csv`
reports both the bound and how many fields it applies to.

The `span_verify_norm` arm applies a deterministic date parser to the selected
span. This recovers the strict metric, and it **weakens (P1)**: the emitted string
is no longer a document substring, only a pure deterministic function of a
grounded span. Provenance survives; the substring property does not. That is
stated wherever the arm appears rather than being quietly enjoyed.

### 4.1 Duplicates

Invoices repeat values — "Total Due" and "Amount Due" both print the total; the
invoice id reprints on every continuation page. A grounding metric insisting on
one canonical location would penalise a correct read, so each field's truth
carries every canonically-equivalent occurrence and any of them counts. The
occurrences are found by *scanning the finished document*, not by bookkeeping
during layout, which catches the coincidences bookkeeping would miss.

## 5. Data, and why it is generated

`gdx.data` places every token, so it knows the exact token index range each
field's value was written at. No annotated real dataset records that. FUNSD, CORD
and SROIE annotate *values*; on them, grounding can only be approximated by string
matching, and an approximate oracle is not an oracle — a model that reads the
right value from the wrong place scores as correct. That is the reason the shipped
results are synthetic, and it is why `gdx.data.real` exists but contributes no
committed number.

The difficulty is not incidental. Each of these is a config switch so its
contribution can be measured rather than assumed:

| knob | what it breaks |
|---|---|
| `distractor_prob` | "Balance Forward", "Amount Paid", "Shipping" carry amounts next to the target; a bottom-right-most-number rule picks one |
| `duplicate_total_prob` | "Amount Due" reprints the total under a label that is *also* a synonym for total |
| `missing_field_prob` | a dropped field makes abstention correct and emitting anything an error |
| `multi_page_prob` | the summary block lands on the last page, far from the header |
| `box_jitter`, `rotation_deg` | row membership stops being an equality on `y` |
| `verbose_date_prob` | the normalisation case selection cannot win |

Amount arithmetic is done in **integer cents**, so `subtotal + tax = total` is
exact and the verification loop's arithmetic check tests a real property.

> **The limitation, stated plainly.** This is a synthetic benchmark. It validates
> a *mechanism* — can a selection head find the right span, and does verification
> catch inconsistency? — and says nothing about how these methods rank on real
> scanned invoices with real OCR errors. Reading order here is the order the
> generator wrote in; real OCR order is noisier. Both are limitations of the
> evidence, not of the argument, and neither is hidden.

## 6. The LLM arm

`gdx.llm` is a provider-agnostic layer: an abstract `LLMClient`, HTTP
implementations for OpenAI-compatible and Anthropic endpoints constructed from
environment variables, and a **deterministic offline stub that is the default**.
The stub needs no key and no network, and every shipped LLM number comes from it.

The stub is a template extractor over the *linearised* token stream — the form a
document reaches a text-only model in. Its failure modes are emergent from that
setup rather than injected:

1. *Layout blindness.* The stream has no geometry, so a trailing "Amount Due"
   captures the `total` anchor.
2. *Normalisation.* It rewrites dates to ISO, which is what a model asked for a
   date does. The result is correct and **appears nowhere in the document** — a
   genuine hallucination under (P1), produced by being helpful.
3. *Arithmetic fill-in.* With no `total` label it computes `subtotal + tax`. Also
   correct, also absent from the document.

Behaviours 2 and 3 are the interesting ones: they show that "hallucination" as
measured here is not the same thing as "wrong". Both are reported separately.

**The stub is not a measurement of any language model.** Nothing in
`docs/RESULTS.md` claims GPT-4 or Claude behaves like this. The arm exists so the
LLM code path is executable offline and so the *token cost* of an
LLM extraction pipeline has a concrete per-document number. Token counts are
whitespace-based, which understates a real BPE tokeniser on numeric text, so every
token figure is a **lower bound** on what a provider would bill.

## 7. Efficiency: the axis, and why it is asymptotic

Params and MACs are inputs; wall-clock is the claim. The axis this method wins on
is **decode cost**, and it is structural:

* A span head emits two indices. One forward pass, `O(1)` sequential steps,
  independent of how long the value is.
* A character decoder emits `L` characters: `L` sequential GRU steps *per field*,
  each with an attention pass over the document. `O(F · L)` sequential steps.

So the gap grows with value length, and `decode_scaling` fits the exponent from
measurements rather than asserting it. The expectation, stated before measuring:
`b ≈ 1` for generation, `b ≈ 0` for selection.

MACs-per-ms is reported alongside, because a MAC reduction does not convert to
wall-clock one-for-one and quoting the MAC ratio alone is how an efficiency claim
becomes misleading.

## 8. References

* Devlin, Chang, Lee & Toutanova. *BERT.* NAACL 2019. — extractive span heads.
* Bahdanau, Cho & Bengio. *Neural machine translation by jointly learning to align
  and translate.* ICLR 2015. — the attentional decoder the baseline uses.
* Raffel et al. *Exploring the limits of transfer learning with a unified
  text-to-text transformer (T5).* JMLR 2020. — relative-position attention bias.
* Tancik et al. *Fourier features let networks learn high frequency functions in
  low dimensional domains.* NeurIPS 2020. — the 2-D positional expansion.
* Xu et al. *LayoutLM: pre-training of text and layout for document image
  understanding.* KDD 2020. — the text-plus-2-D-position input formulation.
* Kim et al. *OCR-free document understanding transformer (Donut).* ECCV 2022. —
  the generative extraction approach this repository's baseline stands for.
* Biten et al. *Scene text visual question answering.* ICCV 2019. — ANLS.
* Jaume, Ekenel & Thiran. *FUNSD.* ICDAR-OST 2019.
* Park et al. *CORD: a consolidated receipt dataset for post-OCR parsing.*
  NeurIPS Workshop 2019.
* Huang et al. *ICDAR 2019 competition on scanned receipt OCR and information
  extraction (SROIE).* ICDAR 2019.
* Guo, Pleiss, Sun & Weinberger. *On calibration of modern neural networks.*
  ICML 2017. — ECE.
* Nixon et al. *Measuring calibration in deep learning.* CVPR Workshops 2019. —
  adaptive/equal-mass calibration error.
* Geifman & El-Yaniv. *Selective classification for deep neural networks.*
  NeurIPS 2017. — risk-coverage and AURC.
* Holm. *A simple sequentially rejective multiple test procedure.* Scandinavian
  Journal of Statistics 1979.
* Wilcoxon. *Individual comparisons by ranking methods.* Biometrics Bulletin 1945.
* Efron & Tibshirani. *An introduction to the bootstrap.* 1993.
