"""Optional real-dataset loaders. **No committed number uses this path.**

FUNSD, CORD and SROIE give word-level text and boxes plus value annotations, so
field accuracy is measurable on them. What they do *not* give is the **token span
each value was read from**, which is the oracle this repository's grounding
metric needs. On real data, grounding can only be approximated by string
matching, and an approximate oracle is not an oracle -- a model that reads the
right value from the wrong place scores as correct.

That is the honest reason the shipped results are synthetic, and it is stated here
rather than in a footnote. The loader exists so a reader can run the method on
real documents and see field accuracy; it is not a substitute for the generator,
and ``docs/RESULTS.md`` contains no number from it.

Nothing here downloads anything. ``scripts/download_real.py`` prints the URLs and
licence terms; the loader reads whatever is already on disk and raises a clear
error otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

from gdx.data.schema import FIELDS, Document, FieldTruth, Token, canonical_value

SUPPORTED = ("funsd", "cord", "sroie")

#: How each dataset's own label vocabulary maps onto this project's field schema.
#: Partial by necessity: FUNSD annotates generic question/answer pairs rather than
#: invoice fields, so only what maps is used and the rest is dropped.
FIELD_ALIASES: dict[str, dict[str, str]] = {
    "cord": {
        "total.total_price": "total",
        "sub_total.subtotal_price": "subtotal",
        "sub_total.tax_price": "tax",
    },
    "sroie": {
        "total": "total",
        "date": "invoice_date",
        "company": "vendor_name",
    },
    "funsd": {},
}


class RealDatasetUnavailable(RuntimeError):
    """Raised when a real dataset is requested but not present on disk."""


def dataset_root(root: str | Path, name: str) -> Path:
    """Expected directory for a dataset, checked for existence.

    Raises:
        ValueError: On an unsupported dataset name.
        RealDatasetUnavailable: If the directory is missing, with the command
            that explains how to obtain it.
    """
    if name not in SUPPORTED:
        raise ValueError(f"unsupported real dataset {name!r}; expected one of {SUPPORTED}")
    path = Path(root) / name
    if not path.exists():
        raise RealDatasetUnavailable(
            f"{path} not found. Run `python scripts/download_real.py --dataset {name}` "
            "for the source URL and licence terms; this repository never downloads "
            "anything automatically."
        )
    return path


def _normalise_box(box: list[float], width: float, height: float) -> tuple[float, ...]:
    """Map a pixel box to normalised page coordinates, clamped to the page."""
    if width <= 0 or height <= 0:
        raise ValueError(f"page size must be positive, got {width}x{height}")
    x0, y0, x1, y1 = (float(v) for v in box[:4])
    out = (x0 / width, y0 / height, x1 / width, y1 / height)
    return tuple(min(1.0, max(0.0, v)) for v in out)


def load_jsonl_documents(path: Path, dataset: str, limit: int | None = None) -> list[Document]:
    """Read a pre-converted JSONL dump into :class:`Document` objects.

    Each line must hold ``tokens`` (list of ``{text, box, page}``), ``width``,
    ``height`` and ``fields`` (field name to value string). Provenance spans are
    **recovered by string matching**, and every field so recovered is marked
    ``requires_normalisation=False`` only when the match was literal -- so the
    approximation is visible in the data rather than assumed away.

    Raises:
        RealDatasetUnavailable: If the file is missing.
    """
    if not path.exists():
        raise RealDatasetUnavailable(f"{path} not found; see scripts/download_real.py")
    docs: list[Document] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if limit is not None and len(docs) >= limit:
            break
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        width = float(payload.get("width", 1.0))
        height = float(payload.get("height", 1.0))
        tokens = [
            Token(
                text=str(tok["text"]),
                box=_normalise_box(tok["box"], width, height),
                page=int(tok.get("page", 0)),
            )
            for tok in payload.get("tokens", [])
        ]
        doc = Document(
            doc_id=int(payload.get("id", i)),
            tokens=tokens,
            fields={name: FieldTruth(name) for name in FIELDS},
            n_pages=1 + max((t.page for t in tokens), default=0),
            meta={"source": dataset, "approximate_provenance": True},
        )
        aliases = FIELD_ALIASES.get(dataset, {})
        for raw_name, value in (payload.get("fields") or {}).items():
            name = aliases.get(raw_name, raw_name)
            if name not in FIELDS or not value:
                continue
            doc.fields[name] = _match_field(doc, name, str(value))
        docs.append(doc)
    return docs


def _match_field(doc: Document, name: str, value: str) -> FieldTruth:
    """Recover a provenance span by canonical string matching, or mark it absent.

    The returned truth records ``present=False`` when no span matches, which is
    the honest outcome: an annotated value the tokens do not contain cannot be
    extracted by selection at all, and pretending otherwise would let a real-data
    evaluation report an impossible ceiling as a model failure.
    """
    target = canonical_value(name, value)
    if not target:
        return FieldTruth(name)
    n = len(doc.tokens)
    for start in range(n):
        page = doc.tokens[start].page
        for end in range(start, min(start + 6, n)):
            if doc.tokens[end].page != page:
                break
            text = doc.span_text(start, end)
            if canonical_value(name, text) == target:
                from gdx.data.schema import normalise_text

                return FieldTruth(
                    name=name,
                    value=value,
                    present=True,
                    span=(start, end),
                    span_text=text,
                    requires_normalisation=normalise_text(text) != normalise_text(value),
                )
    return FieldTruth(name)


def load_real_dataset(
    name: str, root: str | Path = "data/raw", split: str = "test", limit: int | None = None
) -> list[Document]:
    """Load ``<root>/<name>/<split>.jsonl``.

    Args:
        name: One of :data:`SUPPORTED`.
        root: Parent directory.
        split: File stem.
        limit: Maximum documents.

    Raises:
        RealDatasetUnavailable: If the dataset or split file is absent.
    """
    base = dataset_root(root, name)
    return load_jsonl_documents(base / f"{split}.jsonl", name, limit=limit)
