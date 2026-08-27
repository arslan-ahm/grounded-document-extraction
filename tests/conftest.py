"""Shared fixtures. Every fixture is deterministic and needs no network.

The document fixtures are module- or session-scoped because generation is the
most expensive thing the fast test suite does, and ``pytest -m "not slow"`` has to
finish in about two minutes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(2)

from gdx.config import Config, DataConfig  # noqa: E402
from gdx.data.dataset import DocumentDataset, build_splits  # noqa: E402
from gdx.data.featurise import build_vocabulary  # noqa: E402
from gdx.data.generator import generate_dataset, generate_document  # noqa: E402


@pytest.fixture(scope="session")
def data_cfg() -> DataConfig:
    """Default generator configuration."""
    return DataConfig()


@pytest.fixture(scope="session")
def docs(data_cfg: DataConfig):  # noqa: ANN201
    """80 documents at the default difficulty."""
    return generate_dataset(80, data_cfg, seed=0)


@pytest.fixture(scope="session")
def doc(docs):  # noqa: ANN001, ANN201
    """One document."""
    return docs[0]


@pytest.fixture(scope="session")
def vocab():  # noqa: ANN201
    return build_vocabulary(512)


@pytest.fixture
def tiny_cfg() -> Config:
    """A configuration small enough to train inside a test."""
    cfg = Config()
    cfg.data.n_train = 24
    cfg.data.n_val = 8
    cfg.data.n_test = 8
    cfg.data.max_tokens = 160
    cfg.model.d_model = 32
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.model.d_ff = 48
    cfg.model.dec_hidden = 32
    cfg.model.dec_max_len = 10
    cfg.optim.epochs = 1
    cfg.optim.batch_size = 8
    cfg.run.name = "pytest"
    return cfg


@pytest.fixture
def tiny_splits(tiny_cfg: Config, vocab):  # noqa: ANN001, ANN201
    return build_splits(tiny_cfg.data, tiny_cfg.model, vocab, seed=0)


@pytest.fixture
def tiny_dataset(docs, vocab) -> DocumentDataset:  # noqa: ANN001
    return DocumentDataset(list(docs[:8]), vocab, 12)


@pytest.fixture(scope="session")
def make_doc(data_cfg: DataConfig):  # noqa: ANN201
    """Factory for one document at a given id/seed."""

    def _make(doc_id: int = 0, seed: int = 0, **overrides):  # noqa: ANN001, ANN202
        from dataclasses import replace

        return generate_document(doc_id, replace(data_cfg, **overrides), seed=seed)

    return _make
