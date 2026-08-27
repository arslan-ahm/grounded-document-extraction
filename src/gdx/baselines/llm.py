"""LLM-pipeline extraction arm, on top of :mod:`gdx.llm`.

The arm exists for the *cost* axis. An LLM extraction pipeline pays for the whole
serialised document on every request, once per field, and that is a per-document
token bill this module counts exactly. Accuracy from this arm is the offline
stub's accuracy and is labelled as such everywhere it appears; it is not a claim
about any real provider.

The arm has no provenance. It receives a string back and has nowhere to look it
up, which is precisely the structural situation this repository argues against --
so its candidates carry ``span=None`` and the verification loop can only accept or
abstain on them.
"""

from __future__ import annotations

from gdx.data.schema import FIELDS, Document
from gdx.extract import Candidate
from gdx.llm.client import LLMClient, Usage
from gdx.llm.stub import build_prompt


def extract_candidates(
    doc: Document, client: LLMClient, top_k: int = 1
) -> tuple[dict[str, list[Candidate]], Usage]:
    """One request per field. Returns candidates and the usage they cost.

    ``top_k`` is accepted and ignored beyond 1: a single-sample LLM call yields
    one string. Producing a ranked set would need ``n`` samples or logprobs, and
    both multiply the token bill that is the whole point of measuring this arm.
    """
    del top_k
    usage = Usage()
    out: dict[str, list[Candidate]] = {}
    tokens = doc.texts
    for name in FIELDS:
        prompt = build_prompt(tokens, name)
        reply = client.complete(prompt, max_tokens=24)
        usage.add(reply.usage)
        text = (reply.text or "").strip()
        if not text or text.upper() == "NONE":
            out[name] = []
            continue
        # No confidence is available from a text completion, so the arm reports
        # NaN rather than a fabricated 1.0. The calibration table shows NaN for
        # this arm as a consequence, which is the honest outcome.
        out[name] = [Candidate(value=text, prob=float("nan"), span=None)]
    return out, usage
