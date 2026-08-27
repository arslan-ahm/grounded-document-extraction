"""Layout geometry and the IoU maths, against hand-computed values.

IoU is the grounding metric's soft measure, so it gets checked against numbers
worked out by hand rather than against another implementation. The degenerate
cases matter as much as the normal one: a zero-area box, a pair of identical
boxes, and the ``None`` case that must return ``NaN`` rather than 0.0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gdx.data.layout import CHAR_WIDTH, MARGIN, LayoutBuilder, iou


# --- iou -------------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 1.0, 1.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_half_overlap_hand_computed():
    """Two unit squares offset by 0.5 in x: intersection 0.5, union 1.5."""
    a = (0.0, 0.0, 1.0, 1.0)
    b = (0.5, 0.0, 1.5, 1.0)
    assert iou(a, b) == pytest.approx(0.5 / 1.5)


def test_iou_quarter_overlap_hand_computed():
    """Offset 0.5 in both axes: intersection 0.25, union 2 - 0.25 = 1.75."""
    a = (0.0, 0.0, 1.0, 1.0)
    b = (0.5, 0.5, 1.5, 1.5)
    assert iou(a, b) == pytest.approx(0.25 / 1.75)


def test_iou_contained_box_hand_computed():
    """A 0.5 square inside a unit square: intersection 0.25, union 1.0."""
    a = (0.0, 0.0, 1.0, 1.0)
    b = (0.25, 0.25, 0.75, 0.75)
    assert iou(a, b) == pytest.approx(0.25)


def test_iou_disjoint_is_zero():
    assert iou((0.0, 0.0, 0.1, 0.1), (0.9, 0.9, 1.0, 1.0)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)) == 0.0


def test_iou_zero_area_boxes_is_zero_not_nan():
    assert iou((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5)) == 0.0


def test_iou_with_none_is_nan():
    """Undefined must not be scored as a miss."""
    assert math.isnan(iou(None, (0.0, 0.0, 1.0, 1.0)))
    assert math.isnan(iou((0.0, 0.0, 1.0, 1.0), None))
    assert math.isnan(iou(None, None))


def test_iou_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = sorted(rng.random(2)) + sorted(rng.random(2))
        b = sorted(rng.random(2)) + sorted(rng.random(2))
        box_a = (a[0], a[2], a[1], a[3])
        box_b = (b[0], b[2], b[1], b[3])
        assert iou(box_a, box_b) == pytest.approx(iou(box_b, box_a))


def test_iou_is_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = sorted(rng.random(4))
        b = sorted(rng.random(4))
        value = iou((a[0], a[1], a[2], a[3]), (b[0], b[1], b[2], b[3]))
        assert 0.0 <= value <= 1.0


# --- LayoutBuilder ---------------------------------------------------------

def test_write_returns_the_index_range():
    lb = LayoutBuilder()
    assert lb.write(["a", "b", "c"], 0.1, 0.1) == (0, 2)
    assert lb.write(["d"], 0.1, 0.2) == (3, 3)
    assert len(lb.tokens) == 4


def test_write_skips_empty_strings():
    lb = LayoutBuilder()
    span = lb.write(["a", "", "b"], 0.1, 0.1)
    assert span == (0, 1)
    assert [t.text for t in lb.tokens] == ["a", "b"]


def test_write_of_nothing_returns_sentinel():
    """``(-1, -1)`` must not be mistaken for index 0."""
    lb = LayoutBuilder()
    assert lb.write([], 0.1, 0.1) == (-1, -1)
    assert lb.write(["", ""], 0.1, 0.1) == (-1, -1)
    assert lb.tokens == []


def test_boxes_advance_left_to_right():
    lb = LayoutBuilder()
    lb.write(["aa", "bb", "cc"], 0.1, 0.2)
    xs = [t.box[0] for t in lb.tokens]
    assert xs == sorted(xs)
    assert all(t.box[2] > t.box[0] for t in lb.tokens)


def test_box_width_scales_with_text_length_and_size():
    lb = LayoutBuilder()
    lb.write(["aaaa"], 0.1, 0.1, size=1.0)
    lb.write(["aaaa"], 0.1, 0.2, size=2.0)
    narrow, wide = lb.tokens
    assert wide.width == pytest.approx(2 * narrow.width)
    assert narrow.width == pytest.approx(CHAR_WIDTH * 4)


def test_right_aligned_write_ends_at_the_given_edge():
    lb = LayoutBuilder()
    lb.write_right_aligned(["EUR", "1,234.56"], 0.94, 0.5)
    assert lb.tokens[-1].box[2] == pytest.approx(0.94, abs=1e-9)


def test_right_aligned_clamps_at_the_left_margin():
    lb = LayoutBuilder()
    lb.write_right_aligned(["x" * 200], 0.2, 0.5)
    assert lb.tokens[0].box[0] >= MARGIN * 0.5 - 1e-9


def test_new_page_increments_and_tags_tokens():
    lb = LayoutBuilder()
    lb.write(["a"], 0.1, 0.1)
    assert lb.new_page() == 1
    lb.write(["b"], 0.1, 0.1)
    assert [t.page for t in lb.tokens] == [0, 1]
    assert lb.n_pages == 2


def test_apply_noise_with_zero_parameters_is_a_no_op():
    lb = LayoutBuilder()
    lb.write(["a", "b"], 0.1, 0.1)
    before = [t.box for t in lb.tokens]
    lb.apply_noise(0.0, 0.0, np.random.default_rng(0))
    assert [t.box for t in lb.tokens] == before


def test_apply_noise_preserves_text_page_and_count():
    lb = LayoutBuilder()
    lb.write(["a", "b"], 0.1, 0.1)
    lb.new_page()
    lb.write(["c"], 0.1, 0.1)
    lb.apply_noise(0.01, 2.0, np.random.default_rng(0))
    assert [t.text for t in lb.tokens] == ["a", "b", "c"]
    assert [t.page for t in lb.tokens] == [0, 0, 1]


def test_apply_noise_keeps_boxes_valid_and_clamped():
    lb = LayoutBuilder()
    for i in range(30):
        lb.write(["token"], 0.02 * i, 0.03 * i)
    lb.apply_noise(0.05, 8.0, np.random.default_rng(0))
    for tok in lb.tokens:
        x0, y0, x1, y1 = tok.box
        assert 0.0 <= x0 <= x1 <= 1.0
        assert 0.0 <= y0 <= y1 <= 1.0


def test_rotation_is_shared_within_a_page():
    """A page-level rotation must move a whole page coherently.

    If rotation were applied per token it would be indistinguishable from
    jitter, and the generator would not be modelling page skew at all.
    """
    lb = LayoutBuilder()
    # Kept well inside the page: a token at y=0 has its rotated top corner
    # clamped to 0, which would destroy the linear shift field the test looks for.
    for i in range(20):
        lb.write([f"t{i}"], 0.10 + 0.10 * (i % 5), 0.30 + 0.03 * (i // 5))
    before = [t.center for t in lb.tokens]
    lb.apply_noise(0.0, 5.0, np.random.default_rng(3))
    after = [t.center for t in lb.tokens]
    # A rigid rotation about the page centre gives dy = sin(theta) * (x - 0.5),
    # exactly linear in x. Independent per-token noise would not.
    xs = np.array([c[0] for c in before])
    dys = np.array([b[1] - a[1] for a, b in zip(before, after, strict=True)])
    assert dys.std() > 1e-6, "no rotation was applied; the test proved nothing"
    assert abs(np.corrcoef(xs, dys)[0, 1]) > 0.95


def test_token_center_width_height():
    lb = LayoutBuilder()
    lb.write(["abc"], 0.2, 0.4)
    tok = lb.tokens[0]
    assert tok.center[0] == pytest.approx(0.5 * (tok.box[0] + tok.box[2]))
    assert tok.center[1] == pytest.approx(0.5 * (tok.box[1] + tok.box[3]))
    assert tok.width == pytest.approx(tok.box[2] - tok.box[0])
    assert tok.height == pytest.approx(tok.box[3] - tok.box[1])
