"""Page layout: places words as boxes and hands back the token index range.

The generator does not draw pixels. It places *tokens*, and every placement
returns the inclusive ``(start, end)`` index range it occupied, which is what
makes provenance exact. A field's ground-truth span is simply whatever
:meth:`LayoutBuilder.write` returned when its value was written.

Geometry is a deliberately crude monospace model -- character width times a size
scale -- because the research question is about grounding and verification, not
about typography. The two pieces of realism that *are* modelled are the two that
break naive geometric heuristics: per-token box jitter, and a small whole-page
rotation applied to box centres so that "same row" is no longer an exact
equality on ``y``.
"""

from __future__ import annotations

import math

from gdx.data.schema import Token

CHAR_WIDTH = 0.0095
LINE_HEIGHT = 0.026
SPACE_WIDTH = 0.006
MARGIN = 0.07


class LayoutBuilder:
    """Accumulates tokens across pages and reports where each one landed."""

    def __init__(self, page_width: float = 1.0, page_height: float = 1.0) -> None:
        self.tokens: list[Token] = []
        self.page = 0
        self.page_width = page_width
        self.page_height = page_height
        self._page_starts: list[int] = [0]

    # -- page management ----------------------------------------------------

    def new_page(self) -> int:
        """Start a new page and return its index."""
        self.page += 1
        self._page_starts.append(len(self.tokens))
        return self.page

    @property
    def n_pages(self) -> int:
        return self.page + 1

    # -- writing -----------------------------------------------------------

    def write(
        self,
        words: list[str] | tuple[str, ...],
        x: float,
        y: float,
        size: float = 1.0,
    ) -> tuple[int, int]:
        """Place ``words`` left to right starting at ``(x, y)``.

        Args:
            words: Non-empty word strings. Empty strings are skipped so a caller
                can pass a conditionally-built list without guarding it.
            x: Left edge, normalised page coordinate.
            y: Top edge, normalised page coordinate.
            size: Font size multiplier, scaling both glyph width and box height.

        Returns:
            Inclusive ``(start, end)`` token indices. ``(-1, -1)`` if nothing was
            written, which callers must treat as "field absent" rather than as
            index 0.
        """
        kept = [w for w in (str(w) for w in words) if w]
        if not kept:
            return (-1, -1)
        start = len(self.tokens)
        cursor = x
        height = LINE_HEIGHT * size * 0.78
        for word in kept:
            width = max(CHAR_WIDTH * size * len(word), CHAR_WIDTH * size)
            box = (cursor, y, cursor + width, y + height)
            self.tokens.append(Token(text=word, box=box, page=self.page))
            cursor += width + SPACE_WIDTH * size
        return (start, len(self.tokens) - 1)

    def write_right_aligned(
        self,
        words: list[str] | tuple[str, ...],
        right: float,
        y: float,
        size: float = 1.0,
    ) -> tuple[int, int]:
        """Place ``words`` so the last one ends at ``right``.

        Amount columns on invoices are right-aligned, and that alignment is a
        real cue a layout-aware model can use. Emitting everything left-aligned
        would remove a signal that exists in the target domain.
        """
        kept = [w for w in (str(w) for w in words) if w]
        if not kept:
            return (-1, -1)
        total = sum(max(CHAR_WIDTH * size * len(w), CHAR_WIDTH * size) for w in kept)
        total += SPACE_WIDTH * size * (len(kept) - 1)
        return self.write(kept, max(MARGIN * 0.5, right - total), y, size)

    # -- post-processing ---------------------------------------------------

    def apply_noise(self, jitter: float, rotation_deg: float, rng) -> None:  # noqa: ANN001
        """Perturb every box: independent jitter, then a per-page rotation.

        Args:
            jitter: Standard deviation of an independent Gaussian offset added
                to each box corner, in page units.
            rotation_deg: Standard deviation of a per-page rotation angle in
                degrees, applied about the page centre. Applied to the box
                *corners*, so a rotated box widens slightly, exactly as an
                axis-aligned OCR box does on a skewed scan.
            rng: A ``numpy.random.Generator``.

        The order matters: rotation is a page-level property and jitter is a
        token-level one, so jitter must not be rotated as if it were signal.
        """
        if jitter <= 0 and rotation_deg <= 0:
            return
        angles = {
            p: math.radians(float(rng.normal(0.0, rotation_deg)))
            for p in range(self.n_pages)
        }
        out: list[Token] = []
        for tok in self.tokens:
            x0, y0, x1, y1 = tok.box
            if jitter > 0:
                dx0, dy0, dx1, dy1 = rng.normal(0.0, jitter, size=4)
                x0, y0, x1, y1 = x0 + dx0, y0 + dy0, x1 + dx1, y1 + dy1
            theta = angles.get(tok.page, 0.0)
            if theta != 0.0:
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                moved = []
                for cx, cy in corners:
                    rx, ry = cx - 0.5, cy - 0.5
                    moved.append((0.5 + rx * cos_t - ry * sin_t, 0.5 + rx * sin_t + ry * cos_t))
                x0 = min(m[0] for m in moved)
                y0 = min(m[1] for m in moved)
                x1 = max(m[0] for m in moved)
                y1 = max(m[1] for m in moved)
            out.append(
                Token(
                    text=tok.text,
                    box=(
                        _clip(min(x0, x1)),
                        _clip(min(y0, y1)),
                        _clip(max(x0, x1)),
                        _clip(max(y0, y1)),
                    ),
                    page=tok.page,
                )
            )
        self.tokens = out


def _clip(value: float) -> float:
    """Clamp a coordinate into ``[0, 1]``.

    Noise can push a box past the page edge. Clamping keeps every box a valid
    region, which the IoU metric and the positional encoding both assume.
    """
    return float(min(1.0, max(0.0, value)))


def iou(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    """Intersection over union of two axis-aligned boxes.

    Returns ``NaN`` when either box is ``None`` (an undefined comparison must not
    be scored as 0.0, which would be indistinguishable from a real miss) and
    ``0.0`` when both are valid but disjoint or degenerate.
    """
    if a is None or b is None:
        return float("nan")
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)
