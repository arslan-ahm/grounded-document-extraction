"""The central invariant: a grounded extraction cannot emit an absent string.

This is the most important file in the repository. Everything else measures a
rate; this asserts a *guarantee*, and the guarantee is the whole argument. If any
test here fails, the README's headline claim is false and must be retracted.

The invariant is stated three ways, because each catches a different class of bug:

1. **Span text is document text.** Whatever a span head selects, the string it
   yields is by construction a slice of ``doc.tokens``. Tested over many seeds,
   many documents and *adversarially chosen* spans, not only the model's own.
2. **The measurement agrees.** ``mark_grounding`` re-derives grounding by asking
   the document, independently of how the value was produced. It must never come
   back false for a span-derived value.
3. **An untrained model cannot break it.** A model with random weights selects
   nonsense spans. Nonsense spans are still spans, so the hallucination rate is
   still exactly zero -- which is the point: the guarantee does not depend on the
   model being any good.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gdx.arms import ARM_BY_NAME, span_candidates
from gdx.config import Config
from gdx.data.dataset import DocumentDataset
from gdx.data.generator import generate_dataset
from gdx.data.schema import FIELDS, MAX_VALUE_SPAN
from gdx.extract import Candidate, Extraction, mark_grounding
from gdx.models.model import build_model
from gdx.verify import verify_document

SEEDS = (0, 1, 2, 3, 5, 8, 13, 21)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_span_of_every_document_is_present_in_that_document(seed, data_cfg):  # noqa: ANN001
    """Exhaustive: every admissible span's text is found by ``contains_value``.

    This is the invariant at its strongest. It quantifies over *all* spans, not
    the ones a model happened to pick, so it cannot be satisfied by a model that
    is merely conservative.
    """
    docs = generate_dataset(6, data_cfg, seed=seed)
    checked = 0
    for doc in docs:
        n = len(doc.tokens)
        for name in ("invoice_id", "total", "invoice_date", "vendor_name"):
            for start in range(n):
                page = doc.tokens[start].page
                for end in range(start, min(start + MAX_VALUE_SPAN, n)):
                    if doc.tokens[end].page != page:
                        break
                    text = doc.span_text(start, end)
                    ext = Extraction(
                        field_name=name, value=text, span=(start, end), abstained=False
                    )
                    mark_grounding(doc, ext)
                    # A span whose text does not canonicalise for this field's
                    # type (e.g. "Bracket" as an amount) yields "" and is not a
                    # value at all; the invariant is about values.
                    from gdx.data.schema import canonical_value

                    if canonical_value(name, text):
                        assert ext.grounded, (
                            f"span {(start, end)} text {text!r} was not found in its own "
                            f"document for field {name}"
                        )
                        checked += 1
    assert checked > 500, f"only {checked} spans exercised; the test is not covering enough"


@pytest.mark.parametrize("seed", SEEDS)
def test_untrained_span_model_never_hallucinates(seed, data_cfg, vocab):  # noqa: ANN001
    """Random weights, verification off: hallucination rate is exactly 0.

    The guarantee is structural, so it must hold for a model that has learned
    nothing. If this ever failed while the trained model passed, the "guarantee"
    would really be a statistic about a well-trained model.
    """
    cfg = Config()
    cfg.model = replace(cfg.model, head="span", d_model=32, n_layers=1, n_heads=2, d_ff=48)
    docs = generate_dataset(8, data_cfg, seed=seed)
    data = DocumentDataset(docs, vocab, cfg.model.dec_max_len)
    model = build_model(cfg.model, vocab.size, seed=seed)
    arm = ARM_BY_NAME["span_only"]
    verify_cfg = arm.verify_config(cfg.verify)

    emitted = 0
    for batch in data.batches(4):
        for doc, pred in zip(batch.docs, model.predict(batch), strict=True):
            extractions, _ = verify_document(doc, span_candidates(pred), verify_cfg)
            for ext in extractions.values():
                if ext.abstained or not ext.value:
                    continue
                emitted += 1
                assert ext.grounded, f"untrained span head emitted {ext.value!r} not in document"
    assert emitted > 0, "no values emitted; the test proved nothing"


def test_adversarial_candidate_spans_still_ground(doc):  # noqa: ANN001
    """Deliberately wrong spans -- reversed, out of range, cross-page -- stay safe.

    A span selector can be *wrong* in every way and still cannot fabricate. The
    out-of-range and inverted cases must abstain (``span_text`` returns ``""``)
    rather than raise or emit something invented.
    """
    n = len(doc.tokens)
    hostile = [(0, 0), (n - 1, n - 1), (5, 3), (-1, 2), (n - 1, n + 5), (0, n - 1)]
    for span in hostile:
        cands = {name: [Candidate(value="", prob=0.9, span=span)] for name in FIELDS}
        extractions, _ = verify_document(doc, cands, Config().verify)
        for ext in extractions.values():
            if not ext.abstained and ext.value:
                assert ext.grounded, f"span {span} produced ungrounded value {ext.value!r}"


def test_span_text_is_always_a_slice_of_the_token_list(docs):  # noqa: ANN001
    """``span_text`` is literally ``" ".join`` of a token slice, or empty."""
    for doc in docs[:20]:
        n = len(doc.tokens)
        for start in (0, 1, n // 2, n - 2):
            for length in (1, 2, 4):
                end = min(n - 1, start + length - 1)
                text = doc.span_text(start, end)
                if not text:
                    continue
                assert text == " ".join(t.text for t in doc.tokens[start : end + 1])


def test_verification_never_turns_an_abstention_into_a_value(doc):  # noqa: ANN001
    """No candidates in means no value out, under every verification setting."""
    empty = {name: [] for name in FIELDS}
    for enabled in (True, False):
        cfg = replace(Config().verify, enabled=enabled)
        extractions, _ = verify_document(doc, empty, cfg)
        for ext in extractions.values():
            assert ext.abstained
            assert ext.value == ""
            assert ext.span is None
            assert ext.grounded is False


def test_generative_candidate_with_invented_string_is_detected(doc):  # noqa: ANN001
    """The measurement must *catch* a hallucination, not just never see one.

    A test suite that only ever checks the safe path cannot distinguish "nothing
    hallucinates" from "the detector is broken". This injects a string that is
    definitely absent and asserts the detector fires.
    """
    invented = "ZZ-99999999"
    cands = {name: [] for name in FIELDS}
    cands["invoice_id"] = [Candidate(value=invented, prob=0.99, span=None)]
    extractions, _ = verify_document(doc, cands, replace(Config().verify, enabled=False))
    ext = extractions["invoice_id"]
    assert not ext.abstained
    assert ext.value == invented
    assert ext.grounded is False, "the hallucination detector failed to fire"


def test_provenance_check_rejects_the_invented_string(doc):  # noqa: ANN001
    """With verification on, the same invented string is abstained on."""
    cands = {name: [] for name in FIELDS}
    cands["invoice_id"] = [Candidate(value="ZZ-99999999", prob=0.99, span=None)]
    extractions, trace = verify_document(doc, cands, Config().verify)
    assert extractions["invoice_id"].abstained
    assert extractions["invoice_id"].reason == "checks_failed"
    assert trace.provenance_rejections >= 1
