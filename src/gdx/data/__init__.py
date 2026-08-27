"""Synthetic visually-rich document generation with exact provenance truth."""

from __future__ import annotations

from gdx.data.dataset import (
    NULL_INDEX,
    DocumentBatch,
    DocumentDataset,
    Splits,
    build_splits,
)
from gdx.data.featurise import Vocabulary, build_vocabulary, decode_chars, encode_chars
from gdx.data.generator import generate_dataset, generate_document
from gdx.data.layout import iou
from gdx.data.schema import (
    AMOUNT_FIELDS,
    DATE_FIELDS,
    FIELD_INDEX,
    FIELDS,
    Document,
    FieldTruth,
    Token,
    canonical_value,
    normalise_amount,
    normalise_date,
    normalise_text,
    type_matches,
)

__all__ = [
    "AMOUNT_FIELDS",
    "DATE_FIELDS",
    "FIELDS",
    "FIELD_INDEX",
    "NULL_INDEX",
    "Document",
    "DocumentBatch",
    "DocumentDataset",
    "FieldTruth",
    "Splits",
    "Token",
    "Vocabulary",
    "build_splits",
    "build_vocabulary",
    "canonical_value",
    "decode_chars",
    "encode_chars",
    "generate_dataset",
    "generate_document",
    "iou",
    "normalise_amount",
    "normalise_date",
    "normalise_text",
    "type_matches",
]
