"""The generator, and whether its ground truth is actually ground truth.

A generator can produce an impossible or trivial benchmark without any error
surfacing, which is the failure mode these tests exist to prevent. The properties
asserted here are the ones the whole evaluation rests on:

* every present field's recorded span really contains that field's value;
* the arithmetic relation the verification loop checks is *exactly* true in the
  data, so a violation found at inference is a real inconsistency;
* documents fit the token budget, because overflowing it would silently delete
  the summary block and with it the amount ground truth;
* the difficulty knobs actually do something -- a distractor probability that
  changed nothing would make the benchmark easier than it is documented to be.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from gdx.data.generator import DROPPABLE, _budget_items, _money, generate_dataset, generate_document
from gdx.data.schema import (
    AMOUNT_FIELDS,
    DATE_FIELDS,
    FIELDS,
    canonical_value,
    normalise_amount,
    normalise_text,
)

SEEDS = (0, 1, 2, 3, 4)


def test_generation_is_deterministic(data_cfg):  # noqa: ANN001
    a = generate_document(7, data_cfg, seed=3)
    b = generate_document(7, data_cfg, seed=3)
    assert a.texts == b.texts
    assert [t.box for t in a.tokens] == [t.box for t in b.tokens]
    assert {k: v.to_dict() for k, v in a.fields.items()} == {
        k: v.to_dict() for k, v in b.fields.items()
    }


def test_different_ids_give_different_documents(data_cfg):  # noqa: ANN001
    texts = {tuple(generate_document(i, data_cfg, seed=0).texts) for i in range(20)}
    assert len(texts) == 20


def test_different_seeds_give_different_documents(data_cfg):  # noqa: ANN001
    texts = {tuple(generate_document(0, data_cfg, seed=s).texts) for s in range(10)}
    assert len(texts) == 10


@pytest.mark.parametrize("seed", SEEDS)
def test_recorded_span_contains_the_field_value(seed, data_cfg):  # noqa: ANN001
    """The provenance oracle is correct: span text canonically equals the value.

    Except where ``requires_normalisation`` says otherwise, in which case the
    canonical forms must still agree -- a non-ISO date still *denotes* the target.
    """
    for doc in generate_dataset(20, data_cfg, seed=seed):
        for name, truth in doc.fields.items():
            if not truth.present:
                assert truth.span is None
                continue
            assert truth.span is not None
            start, end = truth.span
            assert doc.span_text(start, end) == truth.span_text
            assert canonical_value(name, truth.span_text) == canonical_value(name, truth.value)


@pytest.mark.parametrize("seed", SEEDS)
def test_requires_normalisation_is_exactly_literal_inequality(seed, data_cfg):  # noqa: ANN001
    for doc in generate_dataset(20, data_cfg, seed=seed):
        for truth in doc.fields.values():
            if not truth.present:
                continue
            literal_equal = normalise_text(truth.span_text) == normalise_text(truth.value)
            assert truth.requires_normalisation is (not literal_equal)


@pytest.mark.parametrize("seed", SEEDS)
def test_only_dates_ever_require_normalisation(seed, data_cfg):  # noqa: ANN001
    """Amount targets are the written string, so they are never normalisation cases.

    If this broke, the "cost of selection" table would attribute currency
    formatting to the date limitation.
    """
    for doc in generate_dataset(20, data_cfg, seed=seed):
        for name, truth in doc.fields.items():
            if truth.requires_normalisation:
                assert name in DATE_FIELDS


@pytest.mark.parametrize("seed", SEEDS)
def test_arithmetic_relation_is_exact_in_the_data(seed, data_cfg):  # noqa: ANN001
    """``subtotal + tax == total`` holds to the cent whenever all three are present.

    Integer cents in the generator make this exact. If it were only approximate,
    the verification loop's arithmetic check would fire on correct extractions and
    the ablation would measure a floating-point artefact.
    """
    checked = 0
    for doc in generate_dataset(40, data_cfg, seed=seed):
        values = {n: doc.fields[n] for n in AMOUNT_FIELDS}
        if not all(v.present for v in values.values()):
            continue
        parsed = {n: normalise_amount(v.value) for n, v in values.items()}
        assert parsed["subtotal"] + parsed["tax"] == pytest.approx(parsed["total"], abs=0.005)
        checked += 1
    assert checked > 10


@pytest.mark.parametrize("seed", SEEDS)
def test_line_items_sum_to_the_subtotal(seed, data_cfg):  # noqa: ANN001
    for doc in generate_dataset(20, data_cfg, seed=seed):
        spans = doc.meta["line_item_spans"]
        total = sum(normalise_amount(doc.span_text(s, e)) for s, e in spans)
        assert total == pytest.approx(doc.meta["subtotal_cents"] / 100.0, abs=0.005)


@pytest.mark.parametrize("max_tokens", [128, 160, 192, 256])
def test_generated_documents_fit_the_token_budget(max_tokens, data_cfg):  # noqa: ANN001
    cfg = replace(data_cfg, max_tokens=max_tokens)
    docs = generate_dataset(30, cfg, seed=0)
    assert max(len(d) for d in docs) <= max_tokens
    assert min(len(d) for d in docs) > 20


def test_too_small_a_budget_raises_rather_than_truncating(data_cfg):  # noqa: ANN001
    """Truncating would delete the summary block and the amount ground truth."""
    with pytest.raises(ValueError, match="max_tokens"):
        generate_document(0, replace(data_cfg, max_tokens=40), seed=0)


def test_budget_items_is_monotone_in_max_tokens(data_cfg):  # noqa: ANN001
    values = [_budget_items(replace(data_cfg, max_tokens=m), False) for m in (128, 160, 192, 256)]
    assert values == sorted(values)


@pytest.mark.parametrize("seed", SEEDS)
def test_absent_fields_have_no_span_and_no_value(seed, data_cfg):  # noqa: ANN001
    for doc in generate_dataset(20, data_cfg, seed=seed):
        for name in doc.meta["dropped"]:
            truth = doc.fields[name]
            assert not truth.present
            assert truth.span is None
            assert truth.value == ""
            assert name in DROPPABLE


def test_missing_field_probability_changes_the_absent_rate(data_cfg):  # noqa: ANN001
    """A difficulty knob that did nothing would misdescribe the benchmark."""
    low = generate_dataset(80, replace(data_cfg, missing_field_prob=0.0), seed=0)
    high = generate_dataset(80, replace(data_cfg, missing_field_prob=0.45), seed=0)

    def absent_rate(docs):  # noqa: ANN001, ANN202
        return sum(1 for d in docs for t in d.fields.values() if not t.present) / (
            len(docs) * len(FIELDS)
        )

    assert absent_rate(low) == 0.0
    assert absent_rate(high) > 0.15


def test_verbose_date_probability_changes_the_normalisation_rate(data_cfg):  # noqa: ANN001
    zero = generate_dataset(60, replace(data_cfg, verbose_date_prob=0.0), seed=0)
    one = generate_dataset(60, replace(data_cfg, verbose_date_prob=1.0), seed=0)

    def rate(docs):  # noqa: ANN001, ANN202
        present = [t for d in docs for t in d.fields.values() if t.present]
        return sum(1 for t in present if t.requires_normalisation) / max(1, len(present))

    assert rate(zero) == 0.0
    assert rate(one) > 0.15


def test_distractor_probability_changes_the_token_count(data_cfg):  # noqa: ANN001
    none = generate_dataset(40, replace(data_cfg, distractor_prob=0.0), seed=0)
    many = generate_dataset(40, replace(data_cfg, distractor_prob=1.0), seed=0)
    assert sum(d.meta["n_distractors"] for d in none) == 0
    assert sum(d.meta["n_distractors"] for d in many) > 40


def test_duplicate_total_creates_a_second_occurrence(data_cfg):  # noqa: ANN001
    """"Amount Due" reprints the total, so ``alt_spans`` must record it."""
    docs = generate_dataset(60, replace(data_cfg, duplicate_total_prob=1.0), seed=0)
    with_dup = [d for d in docs if d.fields["total"].present]
    assert with_dup
    assert sum(1 for d in with_dup if d.fields["total"].alt_spans) >= len(with_dup) * 0.8


def test_no_duplicates_when_the_knob_is_off(data_cfg):  # noqa: ANN001
    docs = generate_dataset(40, replace(data_cfg, duplicate_total_prob=0.0), seed=0)
    assert all(not d.meta["duplicate_total"] for d in docs)


def test_multi_page_probability_controls_page_count(data_cfg):  # noqa: ANN001
    single = generate_dataset(40, replace(data_cfg, multi_page_prob=0.0), seed=0)
    multi = generate_dataset(40, replace(data_cfg, multi_page_prob=1.0), seed=0)
    assert {d.n_pages for d in single} == {1}
    assert max(d.n_pages for d in multi) >= 2


def test_max_pages_is_respected(data_cfg):  # noqa: ANN001
    docs = generate_dataset(40, replace(data_cfg, multi_page_prob=1.0, max_pages=2), seed=0)
    assert max(d.n_pages for d in docs) <= 2


def test_boxes_stay_inside_the_page(docs):  # noqa: ANN001
    for doc in docs[:30]:
        for tok in doc.tokens:
            x0, y0, x1, y1 = tok.box
            assert 0.0 <= x0 <= x1 <= 1.0
            assert 0.0 <= y0 <= y1 <= 1.0


def test_pages_are_non_decreasing_in_reading_order(docs):  # noqa: ANN001
    """Page confinement in span decoding relies on this ordering."""
    for doc in docs[:30]:
        pages = [t.page for t in doc.tokens]
        assert pages == sorted(pages)


def test_zero_noise_gives_exact_row_alignment(data_cfg):  # noqa: ANN001
    """With jitter and rotation off, tokens on one row share a ``y``."""
    doc = generate_document(0, replace(data_cfg, box_jitter=0.0, rotation_deg=0.0), seed=0)
    ys = sorted({round(t.box[1], 6) for t in doc.tokens})
    assert len(ys) < len(doc.tokens) / 2


def test_noise_breaks_exact_row_alignment(data_cfg):  # noqa: ANN001
    doc = generate_document(0, replace(data_cfg, box_jitter=0.01, rotation_deg=2.0), seed=0)
    ys = {round(t.box[1], 6) for t in doc.tokens}
    assert len(ys) > len(doc.tokens) * 0.8


@pytest.mark.parametrize(
    ("cents", "style", "expected"),
    [
        (123456, {"symbol": "$", "attached": "1", "thousands": "1"}, ["$1,234.56"]),
        (123456, {"symbol": "$", "attached": "0", "thousands": "0"}, ["$", "1234.56"]),
        (123456, {"symbol": "EUR", "attached": "0", "thousands": "1"}, ["EUR", "1,234.56"]),
        (500, {"symbol": "", "attached": "1", "thousands": "1"}, ["5.00"]),
        (-500, {"symbol": "", "attached": "1", "thousands": "1"}, ["-5.00"]),
    ],
)
def test_money_rendering(cents, style, expected):  # noqa: ANN001
    assert _money(cents, style) == expected


def test_every_document_has_every_field_key(docs):  # noqa: ANN001
    for doc in docs:
        assert set(doc.fields) == set(FIELDS)


def test_total_and_invoice_id_are_never_dropped(docs):  # noqa: ANN001
    """A real invoice always states its own id and its total."""
    for doc in docs:
        assert doc.fields["total"].present
        assert doc.fields["invoice_id"].present
        assert doc.fields["vendor_name"].present
