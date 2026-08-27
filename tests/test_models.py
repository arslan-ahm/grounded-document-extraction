"""Model internals: positional encodings, attention, encoder, both heads.

The attention mechanism and the span head are checked against hand-computed
references rather than against another implementation, because "matches my other
code" is not a correctness argument. The masking behaviour gets particular
attention: a padded position leaking into a softmax is silent, and it would
change every number in the repository by a small unexplained amount.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from gdx.data.dataset import NULL_INDEX
from gdx.data.schema import FIELDS
from gdx.models.attention import NEG_INF, Attention, EncoderLayer
from gdx.models.heads import GenerativeHead, SpanHead, _decode_one
from gdx.models.model import HEADS, GDXModel, build_model
from gdx.models.position import (
    Box2DEncoding,
    Sinusoidal1D,
    SpatialBias,
    bucket_signed_log,
)
from gdx.utils.complexity import attention_macs, count_macs, count_params


# --- Sinusoidal1D ----------------------------------------------------------

def test_sinusoidal_matches_the_closed_form():
    d = 8
    enc = Sinusoidal1D(d, max_len=16)
    table = enc(5)[0]
    for pos in range(5):
        for i in range(0, d, 2):
            angle = pos / (10_000 ** (i / d))
            assert table[pos, i].item() == pytest.approx(math.sin(angle), abs=1e-6)
            assert table[pos, i + 1].item() == pytest.approx(math.cos(angle), abs=1e-6)


def test_sinusoidal_is_fixed_not_learned():
    enc = Sinusoidal1D(8, max_len=16)
    assert list(enc.parameters()) == []


def test_sinusoidal_rejects_odd_width():
    with pytest.raises(ValueError, match="even"):
        Sinusoidal1D(7)


def test_sinusoidal_rejects_too_long_a_sequence():
    enc = Sinusoidal1D(8, max_len=4)
    with pytest.raises(ValueError, match="exceeds max_len"):
        enc(5)


def test_sinusoidal_position_zero_is_alternating_zeros_and_ones():
    row = Sinusoidal1D(8, 4)(1)[0, 0]
    assert torch.allclose(row[0::2], torch.zeros(4), atol=1e-7)
    assert torch.allclose(row[1::2], torch.ones(4), atol=1e-7)


# --- Box2DEncoding ---------------------------------------------------------

def test_box_encoding_output_shape_and_feature_count():
    enc = Box2DEncoding(n_bands=4, d_model=16)
    geom = torch.rand(2, 5, 10)
    assert enc(geom).shape == (2, 5, 16)
    assert enc.n_features == 5 * 2 * 4


def test_box_encoding_rejects_zero_bands():
    with pytest.raises(ValueError, match="n_bands"):
        Box2DEncoding(0, 8)


def test_box_encoding_is_deterministic_and_position_sensitive():
    torch.manual_seed(0)
    enc = Box2DEncoding(6, 12).eval()
    a = torch.zeros(1, 1, 10)
    b = torch.zeros(1, 1, 10)
    b[0, 0, 5] = 0.5  # move the box centre in y
    with torch.no_grad():
        assert torch.allclose(enc(a), enc(a))
        assert not torch.allclose(enc(a), enc(b))


def test_box_encoding_uses_only_the_documented_geometry_slice():
    """Changing x0/y0/x1/y1 alone must not change the output.

    The encoding consumes centre, size and page fraction. If it silently read the
    raw corners too, the ``use_2d_pos`` ablation would remove more than it says.
    """
    torch.manual_seed(0)
    enc = Box2DEncoding(4, 8).eval()
    a = torch.zeros(1, 1, 10)
    b = a.clone()
    b[0, 0, 0:4] = 0.3
    with torch.no_grad():
        assert torch.allclose(enc(a), enc(b))


# --- bucket_signed_log ----------------------------------------------------

def test_bucket_zero_offset_is_the_middle_bucket():
    out = bucket_signed_log(torch.zeros(3), 9)
    assert torch.all(out == 4)


def test_bucket_is_symmetric_about_the_middle():
    deltas = torch.tensor([0.1, -0.1, 0.5, -0.5, 1.0, -1.0])
    out = bucket_signed_log(deltas, 9)
    for i in range(0, 6, 2):
        assert int(out[i]) - 4 == -(int(out[i + 1]) - 4)


def test_bucket_is_monotone_in_magnitude():
    deltas = torch.tensor([0.0, 0.05, 0.2, 0.5, 1.0])
    out = bucket_signed_log(deltas, 11)
    assert list(out) == sorted(out)


def test_bucket_clamps_beyond_max_abs():
    out = bucket_signed_log(torch.tensor([5.0, -5.0]), 9, max_abs=1.0)
    assert int(out[0]) == 8
    assert int(out[1]) == 0


def test_bucket_rejects_too_few_buckets():
    with pytest.raises(ValueError, match="n_buckets"):
        bucket_signed_log(torch.zeros(2), 2)


def test_bucket_makes_even_counts_odd_so_a_zero_bucket_exists():
    out = bucket_signed_log(torch.zeros(1), 10)
    assert int(out[0]) == 4  # 10 -> 9 buckets, middle index 4


# --- SpatialBias -----------------------------------------------------------

def test_spatial_bias_shape():
    bias = SpatialBias(n_heads=3, n_buckets=9)
    geom = torch.rand(2, 6, 10)
    pages = torch.zeros(2, 6, dtype=torch.long)
    assert bias(geom, pages).shape == (2, 3, 6, 6)


def test_spatial_bias_parameter_count_is_small():
    bias = SpatialBias(4, 9)
    assert count_params(bias) == 4 * 9 * 9 + 4 * 2


def test_spatial_bias_diagonal_uses_the_zero_offset_bucket():
    torch.manual_seed(0)
    bias = SpatialBias(1, 9)
    with torch.no_grad():
        bias.table.zero_()
        bias.table[0, 4, 4] = 7.0
        bias.page_bias.zero_()
        geom = torch.zeros(1, 3, 10)
        geom[0, :, 4] = torch.tensor([0.1, 0.5, 0.9])
        geom[0, :, 5] = torch.tensor([0.1, 0.5, 0.9])
        out = bias(geom, torch.zeros(1, 3, dtype=torch.long))
    assert torch.allclose(torch.diagonal(out[0, 0]), torch.full((3,), 7.0))


def test_spatial_bias_distinguishes_pages():
    torch.manual_seed(0)
    bias = SpatialBias(1, 9)
    with torch.no_grad():
        bias.table.zero_()
        bias.page_bias.zero_()
        bias.page_bias[0, 1] = 3.0  # same-page term
        geom = torch.zeros(1, 2, 10)
        pages = torch.tensor([[0, 1]])
        out = bias(geom, pages)
    assert out[0, 0, 0, 0].item() == pytest.approx(3.0)
    assert out[0, 0, 0, 1].item() == pytest.approx(0.0)


# --- Attention -------------------------------------------------------------

def test_attention_shape_and_head_divisibility():
    attn = Attention(16, 4)
    x = torch.rand(2, 5, 16)
    assert attn(x).shape == (2, 5, 16)
    with pytest.raises(ValueError, match="divisible"):
        Attention(10, 4)


def test_attention_weights_are_a_distribution():
    torch.manual_seed(0)
    attn = Attention(16, 4, dropout=0.0).eval()
    x = torch.rand(2, 6, 16)
    mask = torch.ones(2, 6, dtype=torch.bool)
    with torch.no_grad():
        _, weights = attn(x, mask=mask, return_weights=True)
    assert weights.shape == (2, 4, 6, 6)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 6), atol=1e-6)


def test_attention_gives_masked_keys_exactly_zero_weight():
    torch.manual_seed(0)
    attn = Attention(8, 2, dropout=0.0).eval()
    x = torch.rand(1, 5, 8)
    mask = torch.tensor([[True, True, False, False, False]])
    with torch.no_grad():
        _, weights = attn(x, mask=mask, return_weights=True)
    assert torch.all(weights[..., 2:] == 0.0)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 2, 5), atol=1e-6)


def test_attention_output_ignores_masked_content():
    """Changing a padded token's features must not change any output."""
    torch.manual_seed(0)
    attn = Attention(8, 2, dropout=0.0).eval()
    x = torch.rand(1, 5, 8)
    mask = torch.tensor([[True, True, True, False, False]])
    y = x.clone()
    y[0, 3:] = 99.0
    with torch.no_grad():
        a = attn(x, mask=mask)
        b = attn(y, mask=mask)
    assert torch.allclose(a[0, :3], b[0, :3], atol=1e-6)


def test_attention_with_a_fully_masked_row_does_not_produce_nan():
    torch.manual_seed(0)
    attn = Attention(8, 2, dropout=0.0).eval()
    x = torch.rand(1, 4, 8)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    with torch.no_grad():
        out = attn(x, mask=mask)
    assert torch.isfinite(out).all()


def test_attention_bias_shifts_the_logits_as_expected():
    """A large bias on one key must send essentially all mass to it."""
    torch.manual_seed(0)
    attn = Attention(8, 1, dropout=0.0).eval()
    x = torch.rand(1, 4, 8)
    bias = torch.zeros(1, 1, 4, 4)
    bias[0, 0, :, 2] = 50.0
    with torch.no_grad():
        _, weights = attn(x, bias=bias, return_weights=True)
    assert torch.all(weights[0, 0, :, 2] > 0.99)


def test_attention_is_permutation_equivariant_without_position():
    """Attention itself carries no order; the encoder adds it separately."""
    torch.manual_seed(0)
    attn = Attention(8, 2, dropout=0.0).eval()
    x = torch.rand(1, 4, 8)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        a = attn(x)[0][perm]
        b = attn(x[:, perm])[0]
    assert torch.allclose(a, b, atol=1e-6)


def test_encoder_layer_is_residual():
    torch.manual_seed(0)
    layer = EncoderLayer(8, 2, 16, dropout=0.0).eval()
    x = torch.rand(1, 4, 8)
    with torch.no_grad():
        for module in (layer.attn.out_proj, layer.ff[3]):
            module.weight.zero_()
            module.bias.zero_()
        out = layer(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_neg_inf_constant_is_negative_and_finite():
    """A literal ``-inf`` in the logits makes gradients NaN, so a large finite
    sentinel is used instead."""
    assert NEG_INF < -1e8
    assert math.isfinite(NEG_INF)


# --- SpanHead --------------------------------------------------------------

def test_span_head_logit_shapes(tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    head = SpanHead(tiny_cfg.model)
    batch = tiny_dataset.collate([0, 1])
    states = torch.rand(2, batch.seq_len, tiny_cfg.model.d_model)
    start, end = head(states, batch.mask)
    assert start.shape == (2, len(FIELDS), batch.seq_len)
    assert end.shape == start.shape


def test_span_head_masks_padded_positions(tiny_cfg, tiny_dataset):  # noqa: ANN001
    head = SpanHead(tiny_cfg.model)
    batch = tiny_dataset.collate(list(range(len(tiny_dataset))))
    states = torch.rand(len(batch), batch.seq_len, tiny_cfg.model.d_model)
    start, _ = head(states, batch.mask)
    assert torch.all(start[~batch.mask[:, None, :].expand_as(start)] == NEG_INF)


def test_span_head_loss_is_lower_for_the_correct_target(tiny_cfg, tiny_dataset):  # noqa: ANN001
    head = SpanHead(tiny_cfg.model)
    batch = tiny_dataset.collate([0])
    states = torch.rand(1, batch.seq_len, tiny_cfg.model.d_model)
    start, end = head(states, batch.mask)
    good = head.loss(start, end, batch.start, batch.end)
    shifted = torch.clamp(batch.start + 3, max=batch.seq_len - 1)
    bad_start = torch.full_like(batch.start, 1)
    with torch.no_grad():
        start.zero_()
        end.zero_()
        start.scatter_(2, batch.start.unsqueeze(-1), 10.0)
        end.scatter_(2, batch.end.unsqueeze(-1), 10.0)
    aligned = head.loss(start, end, batch.start, batch.end)
    misaligned = head.loss(start, end, bad_start, shifted)
    assert aligned < misaligned
    assert torch.isfinite(good)


# --- _decode_one against hand-computed distributions -----------------------

def _peaked(n: int, index: int, mass: float = 0.9) -> np.ndarray:
    p = np.full(n, (1.0 - mass) / (n - 1))
    p[index] = mass
    return p


def test_decode_one_picks_the_peak():
    n_tokens = 5
    ps = _peaked(n_tokens + 1, 2)
    pe = _peaked(n_tokens + 1, 3)
    pred = _decode_one(ps, pe, n_tokens, 6, "total", 5)
    assert (pred.start, pred.end) == (1, 2)  # sequence indices minus the CLS offset
    assert pred.prob > pred.null_prob


def test_decode_one_abstains_when_the_null_dominates():
    n_tokens = 5
    ps = _peaked(n_tokens + 1, 0, mass=0.98)
    pe = _peaked(n_tokens + 1, 0, mass=0.98)
    pred = _decode_one(ps, pe, n_tokens, 6, "total", 5)
    assert pred.abstained
    assert pred.start == -1


def test_decode_one_respects_max_span_len():
    n_tokens = 10
    ps = _peaked(n_tokens + 1, 1)
    pe = _peaked(n_tokens + 1, 9)
    pred = _decode_one(ps, pe, n_tokens, 3, "total", 5)
    assert pred.start >= 0
    assert pred.end - pred.start + 1 <= 3


def test_decode_one_never_returns_an_inverted_span():
    n_tokens = 8
    ps = _peaked(n_tokens + 1, 7)
    pe = _peaked(n_tokens + 1, 2)
    pred = _decode_one(ps, pe, n_tokens, 6, "total", 5)
    if not pred.abstained:
        assert pred.start <= pred.end
    for s, e, _ in pred.candidates:
        assert s <= e


def test_decode_one_probabilities_are_normalised_over_the_decision_set():
    n_tokens = 4
    rng = np.random.default_rng(0)
    ps = rng.random(n_tokens + 1)
    ps /= ps.sum()
    pe = rng.random(n_tokens + 1)
    pe /= pe.sum()
    pred = _decode_one(ps, pe, n_tokens, 6, "total", 100)
    total = pred.null_prob + sum(p for _, _, p in pred.candidates)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_decode_one_page_mask_forbids_cross_page_spans():
    n_tokens = 6
    pages = np.array([0, 0, 0, 1, 1, 1])
    ps = _peaked(n_tokens + 1, 3)  # token index 2, page 0
    pe = _peaked(n_tokens + 1, 5)  # token index 4, page 1
    pred = _decode_one(ps, pe, n_tokens, 6, "total", 10, pages)
    for s, e, _ in pred.candidates:
        assert pages[s] == pages[e]
    if not pred.abstained:
        assert pages[pred.start] == pages[pred.end]


def test_decode_one_with_zero_tokens_abstains():
    pred = _decode_one(np.array([1.0]), np.array([1.0]), 0, 6, "total", 5)
    assert pred.abstained
    assert pred.candidates == []


def test_decode_one_candidates_are_sorted_by_probability():
    n_tokens = 6
    rng = np.random.default_rng(1)
    ps = rng.random(n_tokens + 1)
    pe = rng.random(n_tokens + 1)
    pred = _decode_one(ps / ps.sum(), pe / pe.sum(), n_tokens, 4, "total", 5)
    probs = [p for _, _, p in pred.candidates]
    assert probs == sorted(probs, reverse=True)


# --- GenerativeHead --------------------------------------------------------

def test_generative_head_teacher_forced_shape(tiny_cfg, tiny_dataset):  # noqa: ANN001
    head = GenerativeHead(tiny_cfg.model)
    batch = tiny_dataset.collate([0, 1])
    states = torch.rand(2, batch.seq_len, tiny_cfg.model.d_model)
    logits = head(states, batch.mask, batch.chars)
    assert logits.shape[:3] == (2, len(FIELDS), batch.chars.shape[2] - 1)


def test_generative_head_decode_returns_strings(tiny_cfg, tiny_dataset):  # noqa: ANN001
    torch.manual_seed(0)
    head = GenerativeHead(tiny_cfg.model).eval()
    batch = tiny_dataset.collate([0, 1])
    states = torch.rand(2, batch.seq_len, tiny_cfg.model.d_model)
    preds = head.decode(states, batch.mask)
    assert len(preds) == 2
    assert [p.field_name for p in preds[0]] == list(FIELDS)
    for pred in preds[0]:
        assert isinstance(pred.text, str)
        assert 0.0 <= pred.prob <= 1.0 or math.isnan(pred.prob)


def test_generative_head_loss_ignores_padding(tiny_cfg, tiny_dataset):  # noqa: ANN001
    head = GenerativeHead(tiny_cfg.model)
    batch = tiny_dataset.collate([0])
    states = torch.rand(1, batch.seq_len, tiny_cfg.model.d_model)
    logits = head(states, batch.mask, batch.chars)
    loss = head.loss(logits, batch.chars)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_generative_head_can_emit_a_string_absent_from_the_document():
    """The property that makes it the reference approach, asserted directly.

    The alphabet is closed but the *language* is not: any string over 70
    characters is reachable, including strings no document contains. That is why
    hallucination is possible for generation and impossible for selection.
    """
    from gdx.data.featurise import CHARSET, decode_chars, encode_chars

    invented = "ZQ-99999999"
    assert all(ch in CHARSET for ch in invented)
    assert decode_chars(encode_chars(invented, 20)) == invented


# --- GDXModel --------------------------------------------------------------

def test_unknown_head_raises(tiny_cfg, vocab):  # noqa: ANN001
    with pytest.raises(ValueError, match="unknown model.head"):
        GDXModel(replace(tiny_cfg.model, head="pointer"), vocab.size)


def test_heads_constant_matches_the_implemented_heads():
    assert set(HEADS) == {"span", "generative"}


@pytest.mark.parametrize("head", HEADS)
def test_model_forward_and_loss(head, tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    model = build_model(replace(tiny_cfg.model, head=head), vocab.size, seed=0)
    batch = tiny_dataset.collate([0, 1])
    out = model.forward(batch)
    assert "states" in out
    loss, logs = model.loss(batch)
    assert torch.isfinite(loss)
    assert set(logs) == {"loss"}
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("head", HEADS)
def test_model_predict_shape(head, tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    model = build_model(replace(tiny_cfg.model, head=head), vocab.size, seed=0)
    batch = tiny_dataset.collate([0, 1, 2])
    preds = model.predict(batch)
    assert len(preds) == 3
    assert all(len(p) == len(FIELDS) for p in preds)


def test_build_model_is_deterministic_in_the_seed(tiny_cfg, vocab):  # noqa: ANN001
    a = build_model(tiny_cfg.model, vocab.size, seed=7)
    b = build_model(tiny_cfg.model, vocab.size, seed=7)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_both_heads_share_the_encoder_initialisation(tiny_cfg, vocab):  # noqa: ANN001
    """The comparison isolates the head, so the encoders must start identical."""
    span = build_model(replace(tiny_cfg.model, head="span"), vocab.size, seed=11)
    gen = build_model(replace(tiny_cfg.model, head="generative"), vocab.size, seed=11)
    for pa, pb in zip(span.encoder.parameters(), gen.encoder.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_predict_restores_training_mode(tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    model.train()
    model.predict(tiny_dataset.collate([0]))
    assert model.training


def test_disabling_2d_pos_removes_the_module_and_parameters(tiny_cfg, vocab):  # noqa: ANN001
    on = build_model(replace(tiny_cfg.model, use_2d_pos=True), vocab.size, seed=0)
    off = build_model(replace(tiny_cfg.model, use_2d_pos=False), vocab.size, seed=0)
    assert on.encoder.embed.box is not None
    assert off.encoder.embed.box is None
    assert count_params(off) < count_params(on)


def test_disabling_spatial_bias_removes_it(tiny_cfg, vocab):  # noqa: ANN001
    off = build_model(replace(tiny_cfg.model, use_spatial_bias=False), vocab.size, seed=0)
    assert off.encoder.spatial is None


def test_span_model_is_smaller_than_the_generative_model(tiny_cfg, vocab):  # noqa: ANN001
    span = build_model(replace(tiny_cfg.model, head="span"), vocab.size, seed=0)
    gen = build_model(replace(tiny_cfg.model, head="generative"), vocab.size, seed=0)
    assert count_params(span) < count_params(gen)


def test_encoder_pool_returns_the_cls_state(tiny_cfg, vocab, tiny_dataset):  # noqa: ANN001
    model = build_model(tiny_cfg.model, vocab.size, seed=0)
    batch = tiny_dataset.collate([0])
    with torch.no_grad():
        states = model.encode(batch)
    assert torch.equal(model.encoder.pool(states), states[:, NULL_INDEX])


# --- complexity accounting -------------------------------------------------

def test_attention_macs_is_quadratic_in_sequence_length():
    a = attention_macs(64, 32, 2)
    b = attention_macs(128, 32, 2)
    assert b == 4 * a


def test_attention_macs_is_linear_in_layers_and_width():
    assert attention_macs(32, 16, 4) == 4 * attention_macs(32, 16, 1)
    assert attention_macs(32, 32, 1) == 2 * attention_macs(32, 16, 1)


def test_count_macs_counts_a_linear_layer_exactly():
    layer = torch.nn.Linear(4, 3)
    assert count_macs(layer, (torch.rand(5, 4),)) == 5 * 4 * 3


def test_count_macs_restores_training_mode():
    layer = torch.nn.Linear(4, 3)
    layer.train()
    count_macs(layer, (torch.rand(2, 4),))
    assert layer.training


def test_count_params_trainable_only():
    layer = torch.nn.Linear(4, 3)
    assert count_params(layer) == 4 * 3 + 3
    layer.bias.requires_grad_(False)
    assert count_params(layer) == 4 * 3
    assert count_params(layer, trainable_only=False) == 4 * 3 + 3
