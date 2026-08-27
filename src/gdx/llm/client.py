"""Provider-agnostic LLM access, with a deterministic offline stub as default.

Every shipped number in this repository comes from :class:`StubClient`. The HTTP
implementations exist so the LLM arm is *real code a reader can point at a real
provider*, not so that anything here needs a key: the default path runs with zero
keys and zero network, and no test touches either.

Token accounting is part of the interface rather than an afterthought. The
efficiency argument against an LLM extraction pipeline is a *cost* argument -- the
prompt has to carry the whole document on every request -- and that cost is only
credible if it is counted per call by the same object that makes the call.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

_WHITESPACE = re.compile(r"\s+")


@dataclass
class Usage:
    """Token accounting for one or many calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


def count_tokens(text: str) -> int:
    """Whitespace token count.

    Deliberately *not* a BPE tokeniser. A real tokeniser is a dependency and a
    download, and both are forbidden on the default path. Whitespace counting
    understates a real BPE count on numeric and punctuation-heavy text -- invoice
    text especially -- so every token-cost figure in this repository is a **lower
    bound** on what a real provider would bill, and is labelled as such.
    """
    text = (text or "").strip()
    return 0 if not text else len(_WHITESPACE.split(text))


@dataclass
class Completion:
    """One model response plus its usage."""

    text: str
    usage: Usage = field(default_factory=Usage)


class LLMClient(ABC):
    """Minimal completion/chat interface."""

    name: str = "abstract"

    def __init__(self, model: str, temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.usage = Usage()

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 64) -> Completion:
        """Single-turn completion."""

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 64) -> Completion:
        """Multi-turn chat, flattened onto :meth:`complete` by default.

        Providers that expose a native chat endpoint override this; the flattening
        is here so a subclass is never *forced* to implement two methods.
        """
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return self.complete(prompt, max_tokens=max_tokens)

    def _record(self, prompt: str, text: str) -> Usage:
        usage = Usage(count_tokens(prompt), count_tokens(text), 1)
        self.usage.add(usage)
        return usage


class HTTPClient(LLMClient):
    """Shared plumbing for the two HTTP providers.

    Uses ``urllib`` from the standard library rather than ``requests`` or a
    vendor SDK: an optional code path must not add a dependency that every reader
    has to install in order to run the default path.
    """

    env_key: str = ""
    default_base: str = ""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(model, temperature)
        self.api_key = api_key or os.environ.get(self.env_key, "")
        self.base_url = (base_url or os.environ.get(f"{self.env_key}_BASE_URL", "")
                         or self.default_base)
        self.timeout = float(timeout)
        if not self.api_key:
            raise RuntimeError(
                f"{type(self).__name__} needs {self.env_key} in the environment; "
                "the default offline stub requires no key"
            )

    def _post(self, url: str, payload: dict, headers: dict[str, str]) -> dict:
        import json
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleClient(HTTPClient):
    """Any OpenAI-compatible ``/chat/completions`` endpoint."""

    name = "openai"
    env_key = "OPENAI_API_KEY"
    default_base = "https://api.openai.com/v1"

    def complete(self, prompt: str, max_tokens: int = 64) -> Completion:
        body = self._post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": max_tokens,
            },
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        text = body["choices"][0]["message"]["content"]
        return Completion(text=text, usage=self._record(prompt, text))


class AnthropicClient(HTTPClient):
    """Anthropic ``/v1/messages`` endpoint."""

    name = "anthropic"
    env_key = "ANTHROPIC_API_KEY"
    default_base = "https://api.anthropic.com/v1"

    def complete(self, prompt: str, max_tokens: int = 64) -> Completion:
        body = self._post(
            f"{self.base_url}/messages",
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        text = "".join(part.get("text", "") for part in body.get("content", []))
        return Completion(text=text, usage=self._record(prompt, text))
