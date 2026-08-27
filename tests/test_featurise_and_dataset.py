"""Featurisation and batching: shapes, dtypes, alignment, and the CLS convention.

The bug this file is mostly guarding against is a *silent misalignment* between
tensors and ground truth. Batching sorts documents by length, so a prediction
attributed to the wrong document would look like a plausible accuracy number
rather than an error. The span-offset convention (``+1`` for the ``[CLS]`` slot)
is the other place a one-off error would be invisible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gdx.data.dataset import NULL_INDEX, DocumentDataset, build_splits
from gdx.data.featurise import (
    CHAR_BOS,
    CHAR_EOS,
    CHAR_PAD,
    CHARSET,
    CLS,
    N_CHARS,
    N_GEOM,
    N_MAG_BUCKETS,
    N_SHAPE_CLASSES,
    PAD,
    SHAPE_INDEX,
    build_vocabulary,
    decode_chars,
    encode_chars,
    encode_document,
    geometry,
    magnitude_bucket,
    shape_of,
)
from gdx.data.schema import FIELDS


# --- vocabulary ------------------------------------------------------------

def test_vocabulary_size_matches_the_request():
    vocab = build_vocabulary(512)
    assert vocab.size == 512
    assert vocab.n_buckets > 16


def test_vocabulary_rejects_a_size_that_starves_the_hash_buckets():
    with pytest.raises(ValueError, match="hash buckets"):
        build_vocabulary(64)


def test_vocabulary_encoding_is_stable_across_calls(vocab):  # noqa: ANN001
    """crc32, not ``hash``: a salted hash would change every process."""
    ids = [vocab.encode(w) for w in ("Bracket", "INV-12345", "9,999.99", "zzz")]
    again = [vocab.encode(w) for w in ("Bracket", "INV-12345", "9,999.99", "zzz")]
    assert ids == again


def test_vocabulary_ids_are_in_range(vocab):  # noqa: ANN001
    for word in ("Bracket", "INV-1", "$5.00", "", "unseen-token-xyz", "Total"):
        idx = vocab.encode(word)
        assert 0 <= idx < vocab.size


def test_closed_words_get_distinct_ids(vocab):  # noqa: ANN001
    a, b = vocab.encode("Bracket"), vocab.encode("Gasket")
    assert a != b
    assert a >= 3 and b >= 3


def test_out_of_vocabulary_words_land_in_the_hash_tail(vocab):  # noqa: ANN001
    idx = vocab.encode("a-string-not-in-the-lexicon-123456")
    assert idx >= 3 + len(vocab.table)


# --- shape classes ---------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2024-01-03", "date_iso"),
        ("03/01/2024", "date_slash"),
        ("3rd", "ordinal"),
        ("March", "month_name"),
        ("Jan", "month_name"),
        ("M12", "measure"),
        ("DN25", "measure"),
        ("INV-12345", "id_code"),
        ("(20%)", "percent"),
        ("1,234.56", "amount"),
        ("$5.00", "amount"),
        ("EUR", "currency"),
        ("$", "currency"),
        ("42", "amount"),
        ("bracket", "alpha_lower"),
        ("Bracket", "alpha_title"),
        ("INVOICE", "alpha_upper"),
        (":", "punct"),
        ("", "pad"),
    ],
)
def test_shape_of_cases(text, expected):  # noqa: ANN001
    assert shape_of(text) == SHAPE_INDEX[expected]


def test_shape_of_is_always_in_range():
    for text in ("", "x", "1", "$1.00", "??", "Mixed1Text", "a" * 50):
        assert 0 <= shape_of(text) < N_SHAPE_CLASSES


# --- magnitude buckets -----------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bracket", 0),
        ("0.05", 1),
        ("0.9", 1),
        ("5", 2),
        ("42", 3),
        ("500.00", 4),
        ("1,234.56", 5),
        ("30,614.90", 6),
    ],
)
def test_magnitude_bucket_cases(text, expected):  # noqa: ANN001
    assert magnitude_bucket(text) == expected


def test_magnitude_bucket_is_monotone_and_bounded():
    values = [magnitude_bucket(str(10**k)) for k in range(0, 12)]
    assert values == sorted(values)
    assert max(values) <= N_MAG_BUCKETS - 1


def test_magnitude_bucket_separates_sub_unit_from_single_digit():
    """Regression: 0.05 and 5.00 must not share a bucket."""
    assert magnitude_bucket("0.05") != magnitude_bucket("5.00")


# --- geometry --------------------------------------------------------------

def test_geometry_shape_and_ranges(doc):  # noqa: ANN001
    geom = geometry(doc)
    assert geom.shape == (len(doc.tokens), N_GEOM)
    assert geom.dtype == np.float32
    assert np.all(geom[:, :8] >= 0.0) and np.all(geom[:, :8] <= 1.0)
    assert np.all((geom[:, 9] == 0.0) | (geom[:, 9] == 1.0))


def test_geometry_centre_and_size_are_consistent(doc):  # noqa: ANN001
    geom = geometry(doc)
    assert np.allclose(geom[:, 4], 0.5 * (geom[:, 0] + geom[:, 2]), atol=1e-6)
    assert np.allclose(geom[:, 5], 0.5 * (geom[:, 1] + geom[:, 3]), atol=1e-6)
    assert np.allclose(geom[:, 6], geom[:, 2] - geom[:, 0], atol=1e-6)


def test_geometry_last_page_flag(docs):  # noqa: ANN001
    multi = next(d for d in docs if d.n_pages > 1)
    geom = geometry(multi)
    for i, tok in enumerate(multi.tokens):
        assert geom[i, 9] == (1.0 if tok.page == multi.n_pages - 1 else 0.0)


def test_encode_document_keys_and_lengths(doc, vocab):  # noqa: ANN001
    enc = encode_document(doc, vocab)
    assert set(enc) == {"word_ids", "shape_ids", "mag_ids", "geom", "page_ids"}
    for key in ("word_ids", "shape_ids", "mag_ids", "page_ids"):
        assert enc[key].shape == (len(doc.tokens),)
        assert enc[key].dtype == np.int64


# --- character coding ------------------------------------------------------

def test_encode_chars_round_trips():
    for value in ("INV-123", "1,234.56", "2024-01-03", "Acme Ltd", ""):
        ids = encode_chars(value, 24)
        assert decode_chars(ids) == value


def test_encode_chars_layout():
    ids = encode_chars("AB", 8)
    assert ids[0] == CHAR_BOS
    assert ids[3] == CHAR_EOS
    assert list(ids[4:]) == [CHAR_PAD] * 4
    assert len(ids) == 8


def test_encode_chars_truncates_without_overflow():
    ids = encode_chars("X" * 100, 10)
    assert len(ids) == 10
    assert CHAR_EOS in list(ids)


def test_encode_chars_drops_characters_outside_the_alphabet():
    """An unknown id would be a token with no surface form."""
    ids = encode_chars("AéB", 12)
    assert decode_chars(ids) == "AB"


def test_decode_chars_stops_at_eos():
    assert decode_chars([CHAR_BOS, 3, 4, CHAR_EOS, 5, 6]) == CHARSET[0] + CHARSET[1]


def test_charset_indices_are_within_the_table():
    assert N_CHARS == 3 + len(CHARSET)
    for value in ("$1,234.56", "INV-9", "2024-12-31"):
        assert all(0 <= int(i) < N_CHARS for i in encode_chars(value, 20))


# --- batching --------------------------------------------------------------

def test_collate_shapes_and_dtypes(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate([0, 1, 2])
    length = batch.seq_len
    assert batch.word_ids.shape == (3, length)
    assert batch.geom.shape == (3, length, N_GEOM)
    assert batch.start.shape == (3, len(FIELDS))
    assert batch.chars.shape == (3, len(FIELDS), tiny_dataset.dec_max_len)
    assert batch.word_ids.dtype == torch.int64
    assert batch.geom.dtype == torch.float32
    assert batch.mask.dtype == torch.bool
    assert batch.present.dtype == torch.float32


def test_cls_slot_is_index_zero_and_always_valid(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate([0, 1])
    assert torch.all(batch.word_ids[:, 0] == CLS)
    assert torch.all(batch.mask[:, 0])
    assert NULL_INDEX == 0


def test_padding_is_masked_out(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate(list(range(len(tiny_dataset))))
    for b, doc in enumerate(batch.docs):
        n = len(doc.tokens)
        assert bool(batch.mask[b, : n + 1].all())
        assert not bool(batch.mask[b, n + 1 :].any())
        assert torch.all(batch.word_ids[b, n + 1 :] == PAD)


def test_span_targets_use_the_plus_one_offset(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate([0, 1, 2])
    for b, doc in enumerate(batch.docs):
        for j, name in enumerate(FIELDS):
            truth = doc.fields[name]
            if truth.present:
                assert int(batch.start[b, j]) == truth.span[0] + 1
                assert int(batch.end[b, j]) == truth.span[1] + 1
                assert float(batch.present[b, j]) == 1.0
            else:
                assert int(batch.start[b, j]) == NULL_INDEX
                assert int(batch.end[b, j]) == NULL_INDEX
                assert float(batch.present[b, j]) == 0.0


def test_targets_point_at_valid_positions(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate(list(range(len(tiny_dataset))))
    for b in range(len(batch)):
        for j in range(len(FIELDS)):
            s, e = int(batch.start[b, j]), int(batch.end[b, j])
            assert bool(batch.mask[b, s]) and bool(batch.mask[b, e])
            assert s <= e


def test_batches_cover_every_document_exactly_once(tiny_dataset):  # noqa: ANN001
    seen = []
    for batch in tiny_dataset.batches(3, shuffle=True, seed=1):
        seen.extend(d.doc_id for d in batch.docs)
    assert sorted(seen) == sorted(d.doc_id for d in tiny_dataset.docs)


def test_batch_docs_align_with_tensor_rows(tiny_dataset):  # noqa: ANN001
    """The alignment that a length-sorting batcher could silently break."""
    for batch in tiny_dataset.batches(3, shuffle=True, seed=2):
        for b, doc in enumerate(batch.docs):
            assert int(batch.mask[b].sum()) == len(doc.tokens) + 1


def test_batch_to_device_is_a_noop_on_cpu(tiny_dataset):  # noqa: ANN001
    batch = tiny_dataset.collate([0])
    moved = batch.to("cpu")
    assert torch.equal(moved.word_ids, batch.word_ids)
    assert moved.docs is batch.docs


def test_max_len_includes_the_cls_slot(tiny_dataset):  # noqa: ANN001
    assert tiny_dataset.max_len == 1 + max(len(d.tokens) for d in tiny_dataset.docs)


def test_empty_dataset_max_len_is_one(vocab):  # noqa: ANN001
    assert DocumentDataset([], vocab, 8).max_len == 1


def test_splits_are_disjoint_by_document_id(tiny_cfg, vocab):  # noqa: ANN001
    splits = build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=0)
    ids = [{d.doc_id for d in s.docs} for s in (splits.train, splits.val, splits.test)]
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]
    assert splits.sizes == {
        "train": tiny_cfg.data.n_train,
        "val": tiny_cfg.data.n_val,
        "test": tiny_cfg.data.n_test,
    }


def test_splits_are_deterministic_in_the_seed(tiny_cfg, vocab):  # noqa: ANN001
    a = build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=5)
    b = build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=5)
    assert [d.texts for d in a.test.docs] == [d.texts for d in b.test.docs]


def test_different_seeds_give_different_populations(tiny_cfg, vocab):  # noqa: ANN001
    a = build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=0)
    b = build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=1)
    assert [d.texts for d in a.test.docs] != [d.texts for d in b.test.docs]
