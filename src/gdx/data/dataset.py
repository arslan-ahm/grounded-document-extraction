"""Splits, tensorisation and batching.

**Why a ``[CLS]`` token at position 0.** The span head has to be able to say
"this field is absent", and the natural way to do that is a dedicated null
position that the start and end distributions can both point at. Putting it at
index 0 rather than at the (batch-dependent) end of the padded sequence means the
null index is the same constant everywhere -- in the loss, in decoding, in the
tests. Every ground-truth span is therefore stored with a ``+1`` offset, and
:data:`NULL_INDEX` is the single place that convention is written down.

**Splits are by document id, not by shuffling a pool.** Documents are a pure
function of ``(doc_id, seed, config)``, so a split is a disjoint id range and
regenerating it costs nothing. Nothing can leak between splits because nothing is
shared between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gdx.config import DataConfig, ModelConfig
from gdx.data.featurise import CLS as CLS_ID
from gdx.data.featurise import N_GEOM, PAD, Vocabulary, encode_chars, encode_document
from gdx.data.generator import generate_dataset
from gdx.data.schema import FIELDS, Document

#: The null position: both span endpoints point here when a field is absent.
NULL_INDEX = 0


@dataclass
class DocumentBatch:
    """A padded batch, plus the documents it came from.

    Carrying the ``Document`` objects along with the tensors is deliberate: the
    hallucination check needs the document's own token strings to ask whether an
    emitted value appears in it, and reconstructing them from ids would be both
    lossy and a place for a bug to hide.
    """

    word_ids: torch.Tensor  # (B, L)
    shape_ids: torch.Tensor  # (B, L)
    mag_ids: torch.Tensor  # (B, L)
    page_ids: torch.Tensor  # (B, L)
    geom: torch.Tensor  # (B, L, N_GEOM)
    mask: torch.Tensor  # (B, L) bool, True where valid (index 0 always True)
    start: torch.Tensor  # (B, F) long
    end: torch.Tensor  # (B, F) long
    present: torch.Tensor  # (B, F) float
    chars: torch.Tensor  # (B, F, T) long
    docs: list[Document]

    def __len__(self) -> int:
        return int(self.word_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.word_ids.shape[1])

    def to(self, device: str | torch.device) -> DocumentBatch:
        return DocumentBatch(
            word_ids=self.word_ids.to(device),
            shape_ids=self.shape_ids.to(device),
            mag_ids=self.mag_ids.to(device),
            page_ids=self.page_ids.to(device),
            geom=self.geom.to(device),
            mask=self.mask.to(device),
            start=self.start.to(device),
            end=self.end.to(device),
            present=self.present.to(device),
            chars=self.chars.to(device),
            docs=self.docs,
        )


class DocumentDataset:
    """A list of documents plus the encoder that turns them into arrays.

    Kept as a plain sequence rather than a ``torch.utils.data.Dataset`` because
    every document fits in memory at this scale and a ``DataLoader`` with worker
    processes on Windows costs more in spawn overhead than it saves.
    """

    def __init__(
        self,
        docs: list[Document],
        vocab: Vocabulary,
        dec_max_len: int = 20,
    ) -> None:
        self.docs = list(docs)
        self.vocab = vocab
        self.dec_max_len = int(dec_max_len)
        self._encoded = [encode_document(d, vocab) for d in self.docs]

    def __len__(self) -> int:
        return len(self.docs)

    def __getitem__(self, index: int) -> tuple[Document, dict[str, np.ndarray]]:
        return self.docs[index], self._encoded[index]

    @property
    def max_len(self) -> int:
        """Longest padded sequence length, including the ``[CLS]`` slot."""
        return 1 + max((len(d.tokens) for d in self.docs), default=0)

    def collate(self, indices: list[int]) -> DocumentBatch:
        """Pad the selected documents into a :class:`DocumentBatch`."""
        chosen = [(self.docs[i], self._encoded[i]) for i in indices]
        length = 1 + max(len(d.tokens) for d, _ in chosen)
        n = len(chosen)
        f = len(FIELDS)

        word_ids = np.full((n, length), PAD, dtype=np.int64)
        shape_ids = np.zeros((n, length), dtype=np.int64)
        mag_ids = np.zeros((n, length), dtype=np.int64)
        page_ids = np.zeros((n, length), dtype=np.int64)
        geom = np.zeros((n, length, N_GEOM), dtype=np.float32)
        mask = np.zeros((n, length), dtype=bool)
        start = np.zeros((n, f), dtype=np.int64)
        end = np.zeros((n, f), dtype=np.int64)
        present = np.zeros((n, f), dtype=np.float32)
        chars = np.zeros((n, f, self.dec_max_len), dtype=np.int64)

        for b, (doc, enc) in enumerate(chosen):
            k = len(doc.tokens)
            word_ids[b, 0] = CLS_ID
            mask[b, 0] = True
            word_ids[b, 1 : k + 1] = enc["word_ids"]
            shape_ids[b, 1 : k + 1] = enc["shape_ids"]
            mag_ids[b, 1 : k + 1] = enc["mag_ids"]
            page_ids[b, 1 : k + 1] = enc["page_ids"]
            geom[b, 1 : k + 1] = enc["geom"]
            mask[b, 1 : k + 1] = True
            for j, name in enumerate(FIELDS):
                truth = doc.fields[name]
                chars[b, j] = encode_chars(truth.value, self.dec_max_len)
                if truth.present and truth.span is not None:
                    start[b, j] = truth.span[0] + 1
                    end[b, j] = truth.span[1] + 1
                    present[b, j] = 1.0
                else:
                    start[b, j] = NULL_INDEX
                    end[b, j] = NULL_INDEX

        return DocumentBatch(
            word_ids=torch.from_numpy(word_ids),
            shape_ids=torch.from_numpy(shape_ids),
            mag_ids=torch.from_numpy(mag_ids),
            page_ids=torch.from_numpy(page_ids),
            geom=torch.from_numpy(geom),
            mask=torch.from_numpy(mask),
            start=torch.from_numpy(start),
            end=torch.from_numpy(end),
            present=torch.from_numpy(present),
            chars=torch.from_numpy(chars),
            docs=[d for d, _ in chosen],
        )

    def batches(
        self, batch_size: int, shuffle: bool = False, seed: int = 0
    ) -> list[DocumentBatch]:
        """All batches for one epoch.

        Documents are sorted by length inside each shuffled chunk so padding
        waste stays low without correlating batch composition with content --
        pure length sorting would put every long multi-page invoice in the same
        batch, which changes the gradient statistics.
        """
        order = np.arange(len(self.docs))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        out: list[DocumentBatch] = []
        for i in range(0, len(order), batch_size):
            chunk = sorted(order[i : i + batch_size].tolist(), key=lambda j: len(self.docs[j]))
            if chunk:
                out.append(self.collate(chunk))
        return out


@dataclass
class Splits:
    """Train / validation / test document sets over disjoint id ranges."""

    train: DocumentDataset
    val: DocumentDataset
    test: DocumentDataset

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def build_splits(
    data: DataConfig,
    model: ModelConfig,
    vocab: Vocabulary,
    seed: int | None = None,
) -> Splits:
    """Generate and encode the three splits.

    ``seed`` shifts the *generator*, so a different seed is a different document
    population as well as a different initialisation. That is the honest thing to
    vary in a seed study: reusing one dataset across seeds would report
    initialisation noise only and understate the true run-to-run scale.
    """
    base = data.seed if seed is None else seed
    n_tr, n_va, n_te = data.n_train, data.n_val, data.n_test
    train = generate_dataset(n_tr, data, seed=base, id_offset=0)
    val = generate_dataset(n_va, data, seed=base, id_offset=n_tr)
    test = generate_dataset(n_te, data, seed=base, id_offset=n_tr + n_va)
    return Splits(
        train=DocumentDataset(train, vocab, model.dec_max_len),
        val=DocumentDataset(val, vocab, model.dec_max_len),
        test=DocumentDataset(test, vocab, model.dec_max_len),
    )
