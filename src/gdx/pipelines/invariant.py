"""Measure the no-hallucination guarantee at scale and write the count.

The guarantee is arithmetic about the output space, not a statistic, so the right
way to report it is an exhaustive count with an explicit denominator: *this many
opportunities to emit an absent string, this many taken*. Three independent
denominators are produced, because each closes a different escape route:

**Enumerated spans.** Every admissible ``(start, end)`` of every document, for
every field, checked against the provenance predicate. This quantifies over the
whole output space rather than over what a model happened to pick, so it cannot
be satisfied by a model that is merely conservative.

**Untrained models.** Random weights over many seeds. If the guarantee held only
for trained models it would be a property of training, not of the interface.

**Trained models.** The shipped checkpoints' actual emissions, which is what a
deployment would see.

The generative arm is measured on the same documents as a control. A denominator
with no positives anywhere would mean the detector was broken rather than that
nothing hallucinates, so the control has to produce a non-zero count.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from gdx.arms import gen_candidates, span_candidates
from gdx.config import Config
from gdx.data.dataset import DocumentDataset
from gdx.data.generator import generate_dataset
from gdx.data.schema import FIELDS, MAX_VALUE_SPAN, canonical_value
from gdx.models.model import build_model
from gdx.pipelines.core import prepare
from gdx.pipelines.experiments import TABLES
from gdx.utils.logging import get_logger
from gdx.verify import verify_document

LOG = get_logger(__name__)


def enumerate_spans(cfg: Config, seeds: tuple[int, ...], n_docs: int = 25) -> dict[str, Any]:
    """Check every admissible span of every document against the predicate."""
    checked = violations = 0
    for seed in seeds:
        for doc in generate_dataset(n_docs, cfg.data, seed=seed):
            n = len(doc.tokens)
            for name in FIELDS:
                for start in range(n):
                    page = doc.tokens[start].page
                    for end in range(start, min(start + MAX_VALUE_SPAN, n)):
                        if doc.tokens[end].page != page:
                            break
                        text = doc.span_text(start, end)
                        if not canonical_value(name, text):
                            continue
                        checked += 1
                        violations += not doc.contains_value(name, text)
    return {
        "population": "all admissible spans, all fields",
        "n_seeds": len(seeds),
        "n_documents": len(seeds) * n_docs,
        "n_opportunities": checked,
        "n_ungrounded": violations,
        "rate": violations / checked if checked else float("nan"),
    }


def _emitted_counts(cfg: Config, head: str, seeds: tuple[int, ...], n_docs: int,
                    trained: bool) -> dict[str, Any]:
    """Count emissions and ungrounded emissions for one head across seeds."""
    from gdx.pipelines.core import train_head

    total = ungrounded = 0
    for seed in seeds:
        prepared = prepare(cfg, seed=seed)
        if trained:
            model, _ = train_head(cfg, prepared, head, None)
            data = prepared.splits.test
        else:
            model = build_model(replace(cfg.model, head=head), prepared.vocab.size, seed=seed)
            data = DocumentDataset(
                generate_dataset(n_docs, cfg.data, seed=seed + 500),
                prepared.vocab,
                cfg.model.dec_max_len,
            )
        for batch in data.batches(cfg.optim.batch_size, shuffle=False):
            for doc, pred in zip(batch.docs, model.predict(batch), strict=True):
                cands = span_candidates(pred) if head == "span" else gen_candidates(pred)
                extractions, _ = verify_document(doc, cands, replace(cfg.verify, enabled=False))
                for ext in extractions.values():
                    if ext.abstained or not ext.value:
                        continue
                    total += 1
                    ungrounded += not ext.grounded
    return {
        "n_seeds": len(seeds),
        "n_opportunities": total,
        "n_ungrounded": ungrounded,
        "rate": ungrounded / total if total else float("nan"),
    }


def measure_invariant(
    cfg: Config,
    untrained_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    trained_seeds: tuple[int, ...] = (0,),
    n_docs: int = 25,
) -> pd.DataFrame:
    """Produce ``results/tables/invariant.csv``.

    Verification is **disabled** for the model rows on purpose. With the loop on,
    a zero would be unremarkable -- the provenance check would have removed the
    offenders. The claim is that selection never produces one in the first place,
    so it has to be measured with the checker switched off.
    """
    rows: list[dict[str, Any]] = []

    row = enumerate_spans(cfg, untrained_seeds, n_docs)
    row["source"] = "enumerated spans (no model)"
    rows.append(row)
    LOG.info("enumerated %d spans, %d ungrounded", row["n_opportunities"], row["n_ungrounded"])

    for head, label in (("span", "untrained span head"), ("generative", "untrained gen. head")):
        row = _emitted_counts(cfg, head, untrained_seeds, n_docs, trained=False)
        row["source"] = label
        row["population"] = f"{n_docs} documents/seed, verification off"
        row["n_documents"] = len(untrained_seeds) * n_docs
        rows.append(row)
        LOG.info("%s: %d emitted, %d ungrounded", label, row["n_opportunities"],
                 row["n_ungrounded"])

    for head, label in (("span", "trained span head"), ("generative", "trained gen. head")):
        row = _emitted_counts(cfg, head, trained_seeds, cfg.data.n_test, trained=True)
        row["source"] = label
        row["population"] = "test split, verification off"
        row["n_documents"] = len(trained_seeds) * cfg.data.n_test
        rows.append(row)
        LOG.info("%s: %d emitted, %d ungrounded", label, row["n_opportunities"],
                 row["n_ungrounded"])

    frame = pd.DataFrame(rows)[
        ["source", "population", "n_seeds", "n_documents", "n_opportunities",
         "n_ungrounded", "rate"]
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / "invariant.csv", index=False)
    return frame
