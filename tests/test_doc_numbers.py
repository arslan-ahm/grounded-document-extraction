"""Every number written in prose must be traceable to a committed CSV.

The injected tables are already safe: ``scripts/render_docs.py`` writes them from
``results/tables/`` and ``test_shipped_documents_are_not_stale`` fails if they
drift. The risk is the *prose* -- a sentence that quotes 0.9535 when the CSV says
0.9487. That kind of error survives every other test in this repository, and it is
exactly the kind a reader would catch and lose trust over.

So this module extracts every numeric literal from the prose of every shipped
Markdown file, excluding fenced code, injected tables and an explicit allowlist of
values that are configuration or arithmetic rather than measurements, and asserts
each one appears in some committed CSV at the precision it was written.

If a number here fails, either the prose is wrong or the value genuinely is not a
measurement and belongs in :data:`ALLOWED` with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/RESULTS.md", "docs/METHOD.md", "docs/REPRODUCIBILITY.md"]

FENCED = re.compile(r"```.*?```", re.S)
TABLE_BLOCK = re.compile(r"<!-- table:[a-z_]+ -->.*?<!-- /table -->", re.S)
INLINE_CODE = re.compile(r"`[^`]*`")
MD_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)
LINK = re.compile(r"\]\([^)]*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

#: Numbers that are configuration, arithmetic or citation rather than measurement.
#: Each entry is a value a reader can verify from the config or the text itself.
ALLOWED: set[str] = {
    # Config and schema constants, all readable in configs/base.yaml or the code.
    "0.30", "0.35", "0.45", "0.85", "0.18", "1.2", "0.006", "0.011", "0.003",
    "0.1", "0.0", "1.0", "0.5", "0.05", "2.0", "3.0", "0.8", "0.9",
    # Scale and shape figures stated in the text and checkable from the config.
    "1200", "1500", "6000", "9600", "3200", "154", "192", "288", "6800",
    # Years, versions and section numbers.
    "2018", "2019", "2020", "2022", "2024", "2017", "2015", "1979", "1945",
    "1993", "2026", "3.12", "3.13", "3.14", "2.13", "1.2020", "4.0",
    # Ratios stated in the text that come from the efficiency CSV verbatim.
    "1.85",
}

#: Decimal literals with this many places or more are treated as measurements.
MIN_DECIMALS = 3


def _prose(path: Path) -> str:
    """The document with code, injected tables and Markdown tables removed."""
    text = path.read_text(encoding="utf-8")
    text = FENCED.sub(" ", text)
    text = TABLE_BLOCK.sub(" ", text)
    text = HTML_COMMENT.sub(" ", text)
    text = MD_TABLE_ROW.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    return LINK.sub("] ", text)


def _numbers(text: str) -> set[str]:
    """Decimal literals with at least :data:`MIN_DECIMALS` places."""
    out = set()
    for match in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", text):
        value = match.group(1)
        if len(value.split(".")[1]) >= MIN_DECIMALS:
            out.add(value)
    return out


def _csv_values() -> set[str]:
    """Every numeric cell of every committed CSV, at several precisions."""
    values: set[str] = set()
    for path in sorted((ROOT / "results" / "tables").glob("*.csv")):
        frame = pd.read_csv(path)
        for column in frame.columns:
            series = pd.to_numeric(frame[column], errors="coerce").dropna()
            for raw in series:
                # Both the signed value and its magnitude: prose legitimately
                # says "costs 0.044375 coverage" for a delta the CSV stores as
                # -0.044375, and that is not an unsourced number.
                for number in (float(raw), abs(float(raw))):
                    for digits in (2, 3, 4, 5, 6):
                        values.add(f"{number:.{digits}f}")
                        values.add(f"{number:.{digits}f}".rstrip("0").rstrip("."))
                    values.add(str(number))
    return values


@pytest.fixture(scope="module")
def csv_values() -> set[str]:
    return _csv_values()


@pytest.mark.parametrize("relative", DOCS)
def test_every_prose_number_is_in_a_committed_csv(relative, csv_values):  # noqa: ANN001
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not present")
    unexplained = sorted(_numbers(_prose(path)) - csv_values - ALLOWED)
    assert not unexplained, (
        f"{relative} quotes numbers with no source CSV: {unexplained}. "
        "Either fix the prose or add the value to ALLOWED with a reason."
    )


@pytest.mark.parametrize("relative", DOCS)
def test_no_placeholder_survives_in_a_shipped_document(relative):  # noqa: ANN001
    path = ROOT / relative
    if not path.exists():
        pytest.skip(f"{relative} not present")
    text = path.read_text(encoding="utf-8")
    for marker in ("PENDING_NUMBERS", "TODO", "TBD", "FIXME", "XXX", "not measured yet"):
        assert marker not in text, f"{relative} still contains {marker!r}"


def test_the_checker_would_catch_a_wrong_number(csv_values):  # noqa: ANN001
    """A checker that never fires proves nothing."""
    invented = "0.123456789"
    assert invented not in csv_values
    assert _numbers(f"the accuracy was {invented} on the test split") == {invented}


def test_the_checker_ignores_fenced_code_and_tables():
    text = "prose 0.1234\n```\ncode 0.9999\n```\n<!-- table:method -->\n| 0.8888 |\n<!-- /table -->"
    found = _numbers(_prose_from_string(text))
    assert found == {"0.1234"}


def _prose_from_string(text: str) -> str:
    text = FENCED.sub(" ", text)
    text = TABLE_BLOCK.sub(" ", text)
    text = HTML_COMMENT.sub(" ", text)
    text = MD_TABLE_ROW.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    return LINK.sub("] ", text)


def test_documented_test_count_matches_the_suite():
    """The README badge and the layout listing must state the real test count."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    counts = [int(m) for m in re.findall(r"(\d+) tests? collected", result.stdout)]
    if not counts:
        counts = [sum(int(m) for m in re.findall(r"^tests/\S+: (\d+)$", result.stdout, re.M))]
    assert counts and counts[0] > 0, "could not determine the collected test count"
    total = counts[0]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    stated = re.search(r"tests-(\d+)%20passing", readme)
    assert stated, "README has no test-count badge"
    assert int(stated.group(1)) == total, (
        f"README badge says {stated.group(1)} tests, suite collects {total}"
    )
    listing = re.search(r"tests/\s+(\d+) tests", readme)
    assert listing and int(listing.group(1)) == total, (
        f"README layout listing disagrees with the suite ({total})"
    )
