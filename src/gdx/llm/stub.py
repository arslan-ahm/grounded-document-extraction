"""The deterministic offline stub. This is what produces every shipped LLM number.

**What it is, stated so no reader can mistake it for something else.** A
template extractor that reads the *serialised token stream* of a document -- the
form a document reaches a text-only LLM in -- and answers one field per call. It
is seeded, has no randomness at inference, needs no key and no network, and its
outputs are reproducible bit for bit.

**What it is not.** It is not a measurement of any real language model. Nothing
in ``docs/RESULTS.md`` claims that GPT-4 or Claude behaves like this. The stub
exists for two purposes: to make the LLM code path executable on the default
offline path, and to give the token-cost accounting a concrete per-document
number.

**Its failure modes are mechanistic, not injected.** Three behaviours are
emergent from "read a linearised stream, answer confidently":

1. *Layout blindness.* The stream has no geometry, so "Amount Due", a synonym for
   ``total``, printed after the real "Total Due", captures the anchor. Distractor
   lines like "Balance Forward" sit in the same neighbourhood.
2. *Normalisation.* It rewrites dates to ISO, because that is what a model asked
   for a date does. The resulting string is correct and **appears nowhere in the
   document** -- a genuine hallucination under the strict definition, produced by
   being helpful.
3. *Arithmetic fill-in.* When ``total`` has no label but ``subtotal`` and ``tax``
   do, it computes the sum. Also correct, also absent from the document.

Behaviours 2 and 3 are the interesting part: they show that "hallucination" as
this repository measures it is not the same thing as "wrong". The results report
both, separately.
"""

from __future__ import annotations

import re

from gdx.data.lexicon import LABEL_SYNONYMS
from gdx.data.schema import DATE_FIELDS, normalise_amount, normalise_date
from gdx.llm.client import Completion, LLMClient

_WORD = re.compile(r"[A-Za-z]+")
FIELD_TAG = re.compile(r"FIELD:\s*([a-z_]+)", re.I)
DOC_TAG = re.compile(r"DOCUMENT:\s*(.*?)\s*FIELD:", re.I | re.S)


def _norm(text: str) -> str:
    return "".join(_WORD.findall(text)).lower()


class StubClient(LLMClient):
    """Offline template extractor over the linearised token stream."""

    name = "stub"

    def __init__(self, model: str = "offline-stub", temperature: float = 0.0) -> None:
        super().__init__(model, temperature)

    def complete(self, prompt: str, max_tokens: int = 64) -> Completion:
        """Answer the ``FIELD:`` request against the ``DOCUMENT:`` body.

        Returns ``"NONE"`` when no label anchor and no type-valid fallback is
        found, which the baseline turns into an abstention. The stub therefore
        *can* abstain -- a stub that never did would make the selective-prediction
        comparison meaningless.
        """
        del max_tokens
        field_match = FIELD_TAG.search(prompt)
        doc_match = DOC_TAG.search(prompt)
        if field_match is None or doc_match is None:
            return Completion(text="NONE", usage=self._record(prompt, "NONE"))
        field_name = field_match.group(1).lower()
        tokens = doc_match.group(1).split()
        text = self._answer(field_name, tokens)
        return Completion(text=text, usage=self._record(prompt, text))

    # -- the extraction rules ---------------------------------------------

    def _answer(self, field_name: str, tokens: list[str]) -> str:
        raw = self._read_after_label(field_name, tokens)
        if raw is None and field_name == "total":
            raw = self._infer_total(tokens)
        if raw is None:
            return "NONE"
        if field_name in DATE_FIELDS:
            iso = normalise_date(raw)
            return iso or raw
        return raw

    def _read_after_label(self, field_name: str, tokens: list[str]) -> str | None:
        """Take the value following the *last* matching label in the stream.

        "Last wins" is the behaviour of a reader that keeps updating its answer
        as it goes, and it is what makes the trailing "Amount Due" duplicate --
        and any distractor printed after the real value -- capture ``total``.
        """
        normed = [_norm(t) for t in tokens]
        synonyms = sorted(
            {_norm(s) for s in LABEL_SYNONYMS.get(field_name, ())}, key=len, reverse=True
        )
        anchor: int | None = None
        n = len(tokens)
        for i in range(n):
            for syn in synonyms:
                if not syn:
                    continue
                joined = ""
                matched = -1
                for j in range(i, min(i + 4, n)):
                    joined += normed[j]
                    if joined == syn:
                        matched = j
                        break
                    if not syn.startswith(joined):
                        break
                if matched >= 0:
                    # Longest synonyms are tried first, so the first match at
                    # position i is the right anchor; taking a shorter one too
                    # would move the anchor back onto "Total" inside "Total Due".
                    anchor = matched
                    break
        if anchor is None:
            return None
        return self._value_at(field_name, tokens, anchor + 1)

    def _value_at(self, field_name: str, tokens: list[str], start: int) -> str | None:
        """First type-valid span of up to four tokens at or after ``start``."""
        from gdx.data.schema import type_matches

        for i in range(start, min(start + 6, len(tokens))):
            for j in range(i, min(i + 4, len(tokens))):
                candidate = " ".join(tokens[i : j + 1])
                if type_matches(field_name, candidate):
                    return candidate
        return None

    def _infer_total(self, tokens: list[str]) -> str | None:
        """``subtotal + tax`` when ``total`` has no label of its own.

        A real hallucination in the strict sense: the emitted digits do not occur
        in the document. Also frequently the right answer. Both facts are
        reported.
        """
        parts = []
        for name in ("subtotal", "tax"):
            raw = self._read_after_label(name, tokens)
            value = normalise_amount(raw or "")
            if value != value:
                return None
            parts.append(value)
        if len(parts) != 2:
            return None
        return f"{parts[0] + parts[1]:.2f}"


def build_prompt(tokens: list[str], field_name: str) -> str:
    """The prompt every LLM-arm call sends. Carries the whole document.

    This is the shape of the cost: one request per field, each request paying for
    the full serialised document. The token-cost table in ``docs/RESULTS.md``
    counts exactly this string.
    """
    return (
        "Extract one field from the invoice below. Reply with the value only, "
        "or NONE if it is absent.\n"
        f"DOCUMENT: {' '.join(tokens)}\n"
        f"FIELD: {field_name}\n"
        "VALUE:"
    )


def build_client(backend: str = "stub", model: str = "offline-stub",
                 temperature: float = 0.0) -> LLMClient:
    """Construct a client by backend name.

    ``"stub"`` is the default everywhere. The HTTP backends raise if their key is
    absent, which is the correct failure: silently falling back to the stub would
    let a reader believe a real provider had been called.
    """
    backend = (backend or "stub").lower()
    if backend == "stub":
        return StubClient(model=model, temperature=temperature)
    if backend in {"openai", "openai-compatible"}:
        from gdx.llm.client import OpenAICompatibleClient

        return OpenAICompatibleClient(model=model, temperature=temperature)
    if backend == "anthropic":
        from gdx.llm.client import AnthropicClient

        return AnthropicClient(model=model, temperature=temperature)
    raise ValueError(f"unknown llm backend {backend!r}; expected stub|openai|anthropic")


__all__ = ["StubClient", "build_client", "build_prompt"]
