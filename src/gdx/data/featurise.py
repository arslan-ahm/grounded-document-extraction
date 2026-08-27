"""Turning a :class:`~gdx.data.schema.Document` into model inputs.

Three feature families, and each exists for a stated reason.

**Word identity, with a closed vocabulary plus hashing.** The lexicon is closed
(:func:`gdx.data.lexicon.build_vocab`) but numeric literals are not, so anything
outside the closed set is hashed into a fixed bucket range. The hash is
``zlib.crc32``, not Python's ``hash``, because ``hash`` on ``str`` is salted per
process: using it would make features depend on which process built them, and
that is a reproducibility bug that does not announce itself.

**Surface shape.** ``INV-81344``, ``2024-01-03``, ``$1,234.56`` and ``Bracket``
belong to different classes regardless of identity, and a model with a 256-bucket
hash cannot recover that from identity alone. This is the feature that lets the
type check and the model agree about what a token *could* be.

**Geometry.** Eight box numbers plus two page numbers. Note that the box goes in
as raw coordinates here; the sinusoidal expansion happens in the model
(:mod:`gdx.models.position`) so that turning 2-D position off is a model-side
switch and does not silently change the data.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass

import numpy as np

from gdx.data.lexicon import build_vocab
from gdx.data.schema import Document, normalise_amount

PAD, UNK, CLS = 0, 1, 2
N_SPECIAL = 3

#: Surface shape classes. Order is fixed forever: it indexes an embedding table.
SHAPE_CLASSES: tuple[str, ...] = (
    "pad",
    "alpha_lower",
    "alpha_title",
    "alpha_upper",
    "digits",
    "amount",
    "date_iso",
    "date_slash",
    "id_code",
    "ordinal",
    "month_name",
    "currency",
    "punct",
    "alnum_mixed",
    "percent",
    "measure",
    "other",
)
N_SHAPE_CLASSES = len(SHAPE_CLASSES)
SHAPE_INDEX = {name: i for i, name in enumerate(SHAPE_CLASSES)}

#: Magnitude buckets for numeric tokens: bucket ``k`` holds values in
#: ``[10^(k-1), 10^k)``. Bucket 0 is "not a number".
N_MAG_BUCKETS = 9

#: Continuous geometry features per token.
N_GEOM = 10

_MONTHS_RE = re.compile(
    r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|"
    r"april|june|july|august|september|october|november|december)$",
    re.I,
)
_ID_RE = re.compile(r"^[A-Za-z]{2,4}[-/]\d{2,8}$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_RE = re.compile(r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}$")
_ORDINAL_RE = re.compile(r"^\d{1,2}(st|nd|rd|th)$", re.I)
_MEASURE_RE = re.compile(r"^(M\d{1,2}|DN\d{1,3}|\d/\d(in|mm)?)$", re.I)


def shape_of(text: str) -> int:
    """Classify a token's surface form into a :data:`SHAPE_CLASSES` index.

    Order of tests matters: ``2024-01-03`` is both "digits and punctuation" and
    an ISO date, and the specific class is the useful one.
    """
    if not text:
        return SHAPE_INDEX["pad"]
    if _ISO_RE.match(text):
        return SHAPE_INDEX["date_iso"]
    if _SLASH_RE.match(text):
        return SHAPE_INDEX["date_slash"]
    if _ORDINAL_RE.match(text):
        return SHAPE_INDEX["ordinal"]
    if _MONTHS_RE.match(text):
        return SHAPE_INDEX["month_name"]
    if _MEASURE_RE.match(text):
        return SHAPE_INDEX["measure"]
    if _ID_RE.match(text):
        return SHAPE_INDEX["id_code"]
    if text.endswith("%") or text.endswith("%)"):
        return SHAPE_INDEX["percent"]
    if normalise_amount(text) == normalise_amount(text) and any(ch.isdigit() for ch in text):
        return SHAPE_INDEX["amount"]
    if text in {"$", "€", "£", "EUR", "GBP", "USD"}:
        return SHAPE_INDEX["currency"]
    if text.isdigit():
        return SHAPE_INDEX["digits"]
    if text.isalpha():
        if text.isupper():
            return SHAPE_INDEX["alpha_upper"]
        if text[0].isupper():
            return SHAPE_INDEX["alpha_title"]
        return SHAPE_INDEX["alpha_lower"]
    if not any(ch.isalnum() for ch in text):
        return SHAPE_INDEX["punct"]
    if any(ch.isdigit() for ch in text) and any(ch.isalpha() for ch in text):
        return SHAPE_INDEX["alnum_mixed"]
    return SHAPE_INDEX["other"]


def magnitude_bucket(text: str) -> int:
    """Log-magnitude bucket of a numeric token, or 0 if it is not numeric.

    Amounts span four orders of magnitude in this generator, and the arithmetic
    relation between subtotal, tax and total is a relation between magnitudes.
    Giving the model the bucket makes that relation representable without it
    having to reconstruct place value from a hashed string identity.
    """
    value = normalise_amount(text)
    if value != value:
        if text.isdigit():
            value = float(text)
        else:
            return 0
    value = abs(value)
    if value < 1.0:
        return 1
    return int(min(N_MAG_BUCKETS - 1, 1 + math.floor(math.log10(value))))


@dataclass
class Vocabulary:
    """Closed word vocabulary plus a fixed hash-bucket tail.

    Attributes:
        table: Word to index, for the closed part.
        n_buckets: Hash buckets for out-of-vocabulary words.
        size: Total embedding rows, including the three special ids.
    """

    table: dict[str, int]
    n_buckets: int

    @property
    def size(self) -> int:
        return N_SPECIAL + len(self.table) + self.n_buckets

    def encode(self, text: str) -> int:
        """Index for one token, hashing anything outside the closed vocabulary."""
        idx = self.table.get(text)
        if idx is not None:
            return N_SPECIAL + idx
        if self.n_buckets <= 0:
            return UNK
        bucket = zlib.crc32(text.encode("utf-8")) % self.n_buckets
        return N_SPECIAL + len(self.table) + bucket


def build_vocabulary(total_size: int = 512) -> Vocabulary:
    """Build the vocabulary, giving all remaining rows to hash buckets.

    Args:
        total_size: Embedding rows. Must leave at least 16 hash buckets after
            the specials and the closed lexicon.

    Raises:
        ValueError: If ``total_size`` is too small, rather than silently
            degrading every numeric token to ``UNK``.
    """
    words = build_vocab()
    table = {word: i for i, word in enumerate(words)}
    n_buckets = total_size - N_SPECIAL - len(table)
    if n_buckets < 16:
        raise ValueError(
            f"model.vocab_size={total_size} leaves {n_buckets} hash buckets for "
            f"{len(table)} closed words; use at least {N_SPECIAL + len(table) + 16}"
        )
    return Vocabulary(table=table, n_buckets=n_buckets)


def geometry(doc: Document) -> np.ndarray:
    """``(L, N_GEOM)`` geometry features: box, centre, size, page position."""
    n = len(doc.tokens)
    out = np.zeros((n, N_GEOM), dtype=np.float32)
    last = max(1, doc.n_pages - 1)
    for i, tok in enumerate(doc.tokens):
        x0, y0, x1, y1 = tok.box
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        out[i] = (
            x0, y0, x1, y1, cx, cy, x1 - x0, y1 - y0,
            tok.page / last,
            1.0 if tok.page == doc.n_pages - 1 else 0.0,
        )
    return out


def encode_document(doc: Document, vocab: Vocabulary) -> dict[str, np.ndarray]:
    """Encode one document's tokens. No padding; the collator does that."""
    texts = doc.texts
    return {
        "word_ids": np.array([vocab.encode(t) for t in texts], dtype=np.int64),
        "shape_ids": np.array([shape_of(t) for t in texts], dtype=np.int64),
        "mag_ids": np.array([magnitude_bucket(t) for t in texts], dtype=np.int64),
        "geom": geometry(doc),
        "page_ids": np.array([t.page for t in doc.tokens], dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Character vocabulary for the generative baseline head
# ---------------------------------------------------------------------------

#: Fixed character set for the generative head. Closed on purpose: the reference
#: approach is free to emit any *string over this alphabet*, which is what makes
#: hallucination possible, but the alphabet itself has to be finite.
CHARSET = " $€£.,-/0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_PAD, CHAR_BOS, CHAR_EOS = 0, 1, 2
CHAR_SPECIAL = 3
CHAR_INDEX = {ch: i + CHAR_SPECIAL for i, ch in enumerate(CHARSET)}
N_CHARS = CHAR_SPECIAL + len(CHARSET)


def encode_chars(value: str, max_len: int) -> np.ndarray:
    """``(max_len,)`` char ids for ``value``: BOS, chars, EOS, then PAD.

    Characters outside :data:`CHARSET` are dropped rather than mapped to a shared
    unknown id, because an unknown id the decoder can emit would be a token with
    no surface form and the hallucination check could not evaluate it.
    """
    ids = [CHAR_BOS]
    for ch in value or "":
        idx = CHAR_INDEX.get(ch)
        if idx is not None:
            ids.append(idx)
        if len(ids) >= max_len - 1:
            break
    ids.append(CHAR_EOS)
    out = np.full((max_len,), CHAR_PAD, dtype=np.int64)
    out[: min(len(ids), max_len)] = ids[:max_len]
    return out


def decode_chars(ids: list[int] | np.ndarray) -> str:
    """Inverse of :func:`encode_chars`, stopping at the first EOS."""
    chars: list[str] = []
    for raw in ids:
        i = int(raw)
        if i in (CHAR_EOS, CHAR_PAD):
            break
        if i == CHAR_BOS:
            continue
        pos = i - CHAR_SPECIAL
        if 0 <= pos < len(CHARSET):
            chars.append(CHARSET[pos])
    return "".join(chars)
