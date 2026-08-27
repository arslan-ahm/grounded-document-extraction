"""The comparison arms, and the code that turns each into candidate sets.

Seven arms, all real, all run in this codebase on the same data with the same
budget and seed:

============================  ==========================================
``heuristic``                 label synonyms + geometric search, best of a
                              six-configuration sweep chosen on validation
``llm_stub``                  the LLM-pipeline arm, offline deterministic stub
``generative``                **the reference approach** -- same encoder, a
                              character decoder, no verification
``generative_verify``         the reference approach *plus* the same
                              verification loop
``span_only``                 selection with verification disabled
``span_verify``               **this repository's method**
``span_verify_norm``          plus deterministic date normalisation
============================  ==========================================

The pair that isolates the claim is ``generative`` versus ``span_only``: one
encoder, one training loop, one seed, one dataset, and *only the output interface
differs*. ``generative_verify`` exists because a fair reading of the argument has
to ask whether the verification loop -- rather than the head -- is what removes
hallucinated values, and the answer is reported rather than avoided.

Model predictions are computed once per split and reused by every arm that shares
a head, so ``span_only``, ``span_verify`` and ``span_verify_norm`` are three
readings of one forward pass. That is not just a speed matter: it guarantees the
three ablations differ in nothing but the switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gdx.baselines.heuristic import HeuristicVariant
from gdx.baselines.heuristic import extract_candidates as heuristic_candidates
from gdx.baselines.llm import extract_candidates as llm_candidates
from gdx.config import Config, VerifyConfig
from gdx.data.dataset import DocumentDataset
from gdx.data.schema import FIELDS
from gdx.extract import Candidate
from gdx.llm.client import LLMClient, Usage
from gdx.models.heads import GenPrediction, SpanPrediction
from gdx.models.model import GDXModel


@dataclass(frozen=True)
class Arm:
    """One comparison arm.

    Attributes:
        name: Table row label.
        source: Where candidates come from -- ``span``, ``generative``,
            ``heuristic`` or ``llm``.
        family: ``ours``, ``baseline``, ``ablation`` or ``reference``.
        verify: Overrides applied to :class:`~gdx.config.VerifyConfig`.
        normalise: Apply deterministic date normalisation after verification.
        note: One-line description carried into the results table.
    """

    name: str
    source: str
    family: str
    verify: dict[str, Any] = field(default_factory=dict)
    normalise: bool = False
    note: str = ""

    def verify_config(self, base: VerifyConfig) -> VerifyConfig:
        """``base`` with this arm's overrides applied."""
        from dataclasses import replace

        return replace(base, **self.verify)


ARMS: tuple[Arm, ...] = (
    Arm(
        "heuristic",
        "heuristic",
        "baseline",
        {"enabled": True},
        note="label synonyms + geometry, best of six on validation",
    ),
    Arm(
        "llm_stub",
        "llm",
        "baseline",
        {"enabled": False},
        note="offline deterministic stub; token-cost arm, not a real provider",
    ),
    Arm(
        "generative",
        "generative",
        "reference",
        {"enabled": False},
        note="the reference approach: same encoder, character decoder",
    ),
    Arm(
        "generative_verify",
        "generative",
        "baseline",
        {"enabled": True},
        note="reference approach plus the same verification loop",
    ),
    Arm(
        "span_only",
        "span",
        "ablation",
        {"enabled": False},
        note="selection, verification disabled",
    ),
    Arm(
        "span_verify",
        "span",
        "ours",
        {"enabled": True},
        note="selection plus the bounded verification loop",
    ),
    Arm(
        "span_verify_norm",
        "span",
        "ours",
        {"enabled": True},
        normalise=True,
        note="plus deterministic date normalisation (weakens the substring guarantee)",
    ),
)

ARM_BY_NAME = {a.name: a for a in ARMS}


def span_candidates(preds: list[SpanPrediction]) -> dict[str, list[Candidate]]:
    """Turn one document's span predictions into ranked candidate sets.

    The abstention decision is *not* baked in here. The head's ranked spans are
    passed through together with the null probability, and the verification loop
    decides. That keeps the abstention policy in one place instead of two.
    """
    out: dict[str, list[Candidate]] = {}
    for pred in preds:
        cands = [
            Candidate(value="", prob=p, span=(s, e))
            for s, e, p in pred.candidates
            if p > pred.null_prob
        ]
        out[pred.field_name] = cands
    return out


def gen_candidates(preds: list[GenPrediction]) -> dict[str, list[Candidate]]:
    """One candidate per field: the greedily decoded string, or none.

    A generative head has no ranked candidate set without a beam, so the
    verification loop can only accept or abstain on these. That asymmetry is real
    and is stated in ``docs/RESULTS.md`` rather than hidden by giving the span
    head a single candidate too -- which would have been the other way to make
    the arms symmetric, and would have thrown away a genuine advantage of
    selection.
    """
    out: dict[str, list[Candidate]] = {}
    for pred in preds:
        out[pred.field_name] = (
            [] if not pred.text else [Candidate(value=pred.text, prob=pred.prob, span=None)]
        )
    return out


def model_candidates(
    model: GDXModel, data: DocumentDataset, cfg: Config, top_k: int = 5
) -> list[dict[str, list[Candidate]]]:
    """Candidate sets for every document in ``data``, in document order.

    Batching reorders documents (short first inside each chunk), so results are
    keyed back by ``doc_id`` and re-sorted. Forgetting that step is a silent
    misalignment between predictions and ground truth, which is the worst class of
    bug this pipeline could have.
    """
    by_id: dict[int, dict[str, list[Candidate]]] = {}
    for batch in data.batches(cfg.optim.batch_size, shuffle=False):
        preds = model.predict(batch, top_k=top_k)
        for doc, pred in zip(batch.docs, preds, strict=True):
            if model.head_name == "span":
                by_id[doc.doc_id] = span_candidates(pred)
            else:
                by_id[doc.doc_id] = gen_candidates(pred)
    return [by_id[d.doc_id] for d in data.docs]


def baseline_candidates(
    source: str,
    data: DocumentDataset,
    variant: HeuristicVariant | None = None,
    client: LLMClient | None = None,
) -> tuple[list[dict[str, list[Candidate]]], Usage]:
    """Candidate sets from a non-neural arm, plus any token usage it cost.

    Raises:
        ValueError: On an unknown source, or a source whose required argument is
            missing -- an LLM arm without a client would otherwise silently
            produce an all-abstain row that looks like a legitimate result.
    """
    usage = Usage()
    if source == "heuristic":
        if variant is None:
            raise ValueError("heuristic arm requires a HeuristicVariant")
        return [heuristic_candidates(doc, variant) for doc in data.docs], usage
    if source == "llm":
        if client is None:
            raise ValueError("llm arm requires an LLMClient")
        out = []
        for doc in data.docs:
            cands, used = llm_candidates(doc, client)
            usage.add(used)
            out.append(cands)
        return out, usage
    raise ValueError(f"unknown baseline source {source!r}")


def empty_candidates(n: int) -> list[dict[str, list[Candidate]]]:
    """``n`` documents' worth of empty candidate sets, for the all-abstain control."""
    return [{name: [] for name in FIELDS} for _ in range(n)]
