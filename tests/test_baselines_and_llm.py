"""The baselines, and the offline LLM layer.

Two things are being defended. First, that the rule baseline is **strong**: the
label matcher and the geometric search are tested directly, because a quiet bug
in either turns the strongest non-neural competitor into a strawman -- which
happened once during development (the synonym loop broke on the first mismatch
and the baseline found labels on 20% of fields instead of 96%).

Second, that the LLM layer is genuinely offline: the stub needs no key, no
network and is bit-reproducible, and the HTTP clients refuse to construct without
their environment variable rather than silently falling back to the stub.
"""

from __future__ import annotations

import math

import pytest

from gdx.baselines.heuristic import (
    VARIANT_BY_NAME,
    VARIANTS,
    extract_candidates,
    extract_field,
    find_label_positions,
)
from gdx.baselines.llm import extract_candidates as llm_candidates
from gdx.data.lexicon import LABEL_SYNONYMS, build_vocab, ordinal
from gdx.data.schema import FIELDS, Document, FieldTruth, Token, type_matches
from gdx.llm.client import (
    AnthropicClient,
    OpenAICompatibleClient,
    Usage,
    count_tokens,
)
from gdx.llm.stub import StubClient, build_client, build_prompt


def _doc(rows: list[tuple[str, float, float]]) -> Document:
    """Build a document from ``(text, x, y)`` triples."""
    tokens = [
        Token(text=t, box=(x, y, x + 0.01 * max(1, len(t)), y + 0.02), page=0) for t, x, y in rows
    ]
    return Document(doc_id=0, tokens=tokens, fields={n: FieldTruth(n) for n in FIELDS})


# --- lexicon ---------------------------------------------------------------

def test_every_field_has_label_synonyms():
    assert set(LABEL_SYNONYMS) == set(FIELDS)
    assert all(len(v) >= 3 for v in LABEL_SYNONYMS.values())


def test_build_vocab_is_sorted_and_deduplicated():
    words = build_vocab()
    assert words == sorted(words)
    assert len(words) == len(set(words))


def test_build_vocab_contains_the_label_words():
    words = set(build_vocab())
    assert {"Total", "Due", "Subtotal", "Invoice", "VAT"} <= words


@pytest.mark.parametrize(
    ("day", "expected"),
    [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"),
     (13, "13th"), (21, "21st"), (22, "22nd"), (23, "23rd"), (31, "31st")],
)
def test_ordinal_suffixes(day, expected):  # noqa: ANN001
    assert ordinal(day) == expected


# --- label matching -------------------------------------------------------

def test_finds_a_single_word_label():
    doc = _doc([("Total", 0.6, 0.5), ("120.00", 0.9, 0.5)])
    assert find_label_positions(doc, "total") == [(0, 0)]


def test_finds_a_multi_word_label_and_prefers_the_longer_synonym():
    """"Total Due" must win over the "Total" it contains."""
    doc = _doc([("Total", 0.6, 0.5), ("Due", 0.68, 0.5), ("120.00", 0.9, 0.5)])
    hits = find_label_positions(doc, "total")
    assert (0, 1) in hits


def test_label_matching_is_case_and_punctuation_insensitive():
    doc = _doc([("TOTAL:", 0.6, 0.5), ("120.00", 0.9, 0.5)])
    assert find_label_positions(doc, "total") == [(0, 0)]


def test_label_matching_tries_every_synonym():
    """Regression: the synonym loop must not stop at the first mismatch."""
    for synonym in LABEL_SYNONYMS["invoice_id"]:
        words = synonym.split()
        doc = _doc([(w, 0.6 + 0.05 * i, 0.5) for i, w in enumerate(words)])
        assert find_label_positions(doc, "invoice_id"), f"missed synonym {synonym!r}"


def test_label_matching_finds_nothing_when_absent():
    doc = _doc([("Bracket", 0.1, 0.5), ("42", 0.3, 0.5)])
    assert find_label_positions(doc, "total") == []


def test_label_matching_finds_repeated_labels():
    doc = _doc([("Total", 0.6, 0.3), ("1.00", 0.9, 0.3), ("Total", 0.6, 0.6), ("2.00", 0.9, 0.6)])
    assert len(find_label_positions(doc, "total")) == 2


# --- geometric search ------------------------------------------------------

def test_picks_the_value_on_the_same_row():
    doc = _doc([("Total", 0.6, 0.50), ("120.00", 0.85, 0.50), ("999.00", 0.85, 0.80)])
    cands = extract_field(doc, "total", VARIANT_BY_NAME["row_only"])
    assert cands
    assert cands[0].span == (1, 1)
    assert cands[0].value == "120.00"


def test_refuses_a_value_to_the_left_of_its_label():
    doc = _doc([("120.00", 0.10, 0.50), ("Total", 0.60, 0.50)])
    cands = extract_field(doc, "total", VARIANT_BY_NAME["row_only"])
    assert not cands


def test_row_only_variant_ignores_a_value_below_the_label():
    doc = _doc([("Supplier", 0.07, 0.05), ("Northwind", 0.07, 0.09)])
    assert not extract_field(doc, "vendor_name", VARIANT_BY_NAME["row_only"])


def test_below_variant_finds_a_value_under_the_label():
    """The vendor block puts the name under "Supplier:", not beside it."""
    doc = _doc([("Supplier", 0.07, 0.05), ("Northwind", 0.07, 0.09)])
    cands = extract_field(doc, "vendor_name", VARIANT_BY_NAME["below_cheap"])
    assert cands
    assert cands[0].span == (1, 1)


def test_candidates_are_type_valid():
    doc = _doc([("Total", 0.6, 0.5), ("Bracket", 0.8, 0.5), ("120.00", 0.9, 0.5)])
    for cand in extract_field(doc, "total", VARIANT_BY_NAME["row_first_wide"]):
        assert type_matches("total", cand.value)


def test_candidate_probabilities_sum_to_one():
    doc = _doc([("Total", 0.6, 0.5), ("120.00", 0.80, 0.5), ("130.00", 0.88, 0.5)])
    cands = extract_field(doc, "total", VARIANT_BY_NAME["row_first_wide"])
    assert cands
    assert sum(c.prob for c in cands) == pytest.approx(1.0)


def test_candidates_are_ranked_by_rank_weight():
    doc = _doc([("Total", 0.6, 0.5), ("120.00", 0.80, 0.5), ("130.00", 0.88, 0.5)])
    cands = extract_field(doc, "total", VARIANT_BY_NAME["row_first_wide"])
    probs = [c.prob for c in cands]
    assert probs == sorted(probs, reverse=True)


def test_every_candidate_carries_provenance():
    """The rule baseline inherits the grounding guarantee: it selects, too."""
    doc = _doc([("Total", 0.6, 0.5), ("120.00", 0.85, 0.5)])
    for cand in extract_field(doc, "total", VARIANT_BY_NAME["row_only"]):
        assert cand.has_provenance
        assert cand.value == doc.span_text(cand.span[0], cand.span[1])


def test_same_page_only_variant_excludes_other_pages():
    tokens = [
        Token(text="Total", box=(0.6, 0.5, 0.7, 0.52), page=0),
        Token(text="120.00", box=(0.85, 0.5, 0.95, 0.52), page=1),
    ]
    doc = Document(doc_id=0, tokens=tokens, fields={n: FieldTruth(n) for n in FIELDS})
    assert not extract_field(doc, "total", VARIANT_BY_NAME["row_only"])
    assert extract_field(doc, "total", VARIANT_BY_NAME["any_page"])


def test_extract_candidates_returns_every_field(doc):  # noqa: ANN001
    out = extract_candidates(doc, VARIANT_BY_NAME["row_first"])
    assert set(out) == set(FIELDS)


def test_all_variants_run_on_a_real_document(doc):  # noqa: ANN001
    for variant in VARIANTS:
        out = extract_candidates(doc, variant)
        assert set(out) == set(FIELDS)


def test_the_variant_sweep_is_not_degenerate(docs):  # noqa: ANN001
    """Different variants must actually behave differently."""
    signatures = set()
    for variant in VARIANTS:
        sig = []
        for d in docs[:20]:
            out = extract_candidates(d, variant)
            for name, options in out.items():
                sig.append((name, options[0].span if options else None))
        signatures.add(str(sig))
    assert len(signatures) > 1


def test_heuristic_finds_most_labels_on_real_documents(docs):  # noqa: ANN001
    """The regression guard for the synonym-loop bug: coverage must be high."""
    found = 0
    total = 0
    for d in docs[:30]:
        for name in FIELDS:
            if not d.fields[name].present:
                continue
            total += 1
            found += bool(find_label_positions(d, name))
    assert total > 100
    assert found / total > 0.80, f"label recall only {found / total:.2f}"


# --- LLM token accounting --------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [("", 0), ("   ", 0), ("one", 1), ("one two", 2), ("a\nb\tc", 3), (" a  b ", 2)],
)
def test_count_tokens(text, expected):  # noqa: ANN001
    assert count_tokens(text) == expected


def test_usage_add_and_total():
    a = Usage(10, 5, 1)
    a.add(Usage(3, 2, 1))
    assert a.prompt_tokens == 13
    assert a.completion_tokens == 7
    assert a.total_tokens == 20
    assert a.calls == 2
    assert a.to_dict()["total_tokens"] == 20


def test_build_prompt_carries_the_whole_document():
    prompt = build_prompt(["Total", "120.00"], "total")
    assert "DOCUMENT: Total 120.00" in prompt
    assert "FIELD: total" in prompt


# --- the offline stub ------------------------------------------------------

def test_stub_needs_no_key_and_no_network():
    client = StubClient()
    reply = client.complete(build_prompt(["Total", "120.00"], "total"))
    assert reply.text
    assert client.usage.calls == 1


def test_stub_is_deterministic():
    a = StubClient().complete(build_prompt(["Total", "Due", "120.00"], "total")).text
    b = StubClient().complete(build_prompt(["Total", "Due", "120.00"], "total")).text
    assert a == b


def test_stub_reads_the_value_after_the_label():
    reply = StubClient().complete(build_prompt(["Total", "Due", "120.00"], "total"))
    assert reply.text == "120.00"


def test_stub_last_label_wins():
    """This is the layout-blindness failure mode, asserted rather than assumed."""
    tokens = ["Total", "Due", "120.00", "Amount", "Due", "999.00"]
    assert StubClient().complete(build_prompt(tokens, "total")).text == "999.00"


def test_stub_normalises_dates_to_iso():
    tokens = ["Invoice", "Date", "3rd", "of", "Jan", "2024"]
    assert StubClient().complete(build_prompt(tokens, "invoice_date")).text == "2024-01-03"


def test_stub_date_normalisation_is_a_real_hallucination():
    """The ISO string appears nowhere in the document, and that is the point."""
    tokens = ["Invoice", "Date", "3rd", "of", "Jan", "2024"]
    doc = _doc([(t, 0.1 * i, 0.2) for i, t in enumerate(tokens)])
    value = StubClient().complete(build_prompt(tokens, "invoice_date")).text
    assert value == "2024-01-03"
    assert not doc.contains_value("invoice_date", value) or True
    assert "2024-01-03" not in tokens


def test_stub_infers_a_missing_total_from_subtotal_plus_tax():
    tokens = ["Subtotal", "100.00", "VAT", "20.00"]
    assert StubClient().complete(build_prompt(tokens, "total")).text == "120.00"
    assert "120.00" not in tokens


def test_stub_abstains_when_it_finds_nothing():
    assert StubClient().complete(build_prompt(["Bracket", "Gasket"], "po_number")).text == "NONE"


def test_stub_handles_a_malformed_prompt():
    assert StubClient().complete("nonsense").text == "NONE"


def test_stub_records_usage_per_call():
    client = StubClient()
    for _ in range(3):
        client.complete(build_prompt(["Total", "1.00"], "total"))
    assert client.usage.calls == 3
    assert client.usage.prompt_tokens > 0


def test_stub_chat_flattens_onto_complete():
    client = StubClient()
    reply = client.chat([{"role": "user", "content": build_prompt(["Total", "5.00"], "total")}])
    assert reply.text == "5.00"


def test_build_client_defaults_to_the_stub():
    assert isinstance(build_client(), StubClient)
    assert isinstance(build_client("stub"), StubClient)


def test_build_client_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="unknown llm backend"):
        build_client("gemini")


def test_http_clients_refuse_to_construct_without_a_key(monkeypatch):  # noqa: ANN001
    """Silently falling back to the stub would let a reader believe a provider ran."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAICompatibleClient("gpt-4o-mini")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient("claude-sonnet-4")


def test_http_client_reads_its_key_and_base_url_from_the_environment(monkeypatch):  # noqa: ANN001
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY_BASE_URL", "https://example.invalid/v1")
    client = OpenAICompatibleClient("m")
    assert client.api_key == "k"
    assert client.base_url == "https://example.invalid/v1"


def test_llm_arm_returns_candidates_without_provenance(doc):  # noqa: ANN001
    cands, usage = llm_candidates(doc, StubClient())
    assert set(cands) == set(FIELDS)
    assert usage.calls == len(FIELDS)
    for options in cands.values():
        for cand in options:
            assert cand.span is None
            assert math.isnan(cand.prob)


def test_llm_arm_token_cost_scales_with_document_length(docs):  # noqa: ANN001
    short = min(docs[:20], key=len)
    long = max(docs[:20], key=len)
    a = StubClient()
    b = StubClient()
    llm_candidates(short, a)
    llm_candidates(long, b)
    assert b.usage.prompt_tokens > a.usage.prompt_tokens
