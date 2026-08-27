"""Inject rendered CSV tables into the documentation between markers.

    python scripts/render_docs.py           # rewrite the docs in place
    python scripts/render_docs.py --check   # fail if any table is stale

A table in ``README.md`` or ``docs/*.md`` is written as::

    <!-- table:method -->
    <!-- /table -->

and this script fills the gap from ``results/tables/`` via :mod:`gdx.report`. No
number in the documentation is typed by hand, so none of them can drift from the
artefact it came from. ``--check`` is what CI and ``tests/test_render_docs.py``
run.
"""

from __future__ import annotations

import argparse
import re

import _bootstrap

from gdx.report import ALL

ROOT = _bootstrap.ROOT
TARGETS = ["README.md", "docs/RESULTS.md", "docs/METHOD.md", "docs/REPRODUCIBILITY.md"]
PATTERN = re.compile(r"(<!-- table:([a-z_]+) -->\n)(.*?)(<!-- /table -->)", re.S)


def render(text: str) -> tuple[str, list[str]]:
    """Return the filled text and the names of tables whose body changed.

    Raises:
        KeyError: On a marker naming a table that does not exist, listing the
            known ones -- a typo in a marker would otherwise leave a permanently
            empty block in the documentation.
    """
    changed: list[str] = []

    def repl(match: re.Match) -> str:
        head, name, body, tail = match.groups()
        if name not in ALL:
            raise KeyError(f"unknown table {name!r}; known: {sorted(ALL)}")
        new_body = ALL[name]() + "\n"
        if new_body != body:
            changed.append(name)
        return f"{head}{new_body}{tail}"

    return PATTERN.sub(repl, text), changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    stale = False
    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        filled, changed = render(text)
        if changed:
            stale = True
            print(f"  {rel}: {', '.join(changed)}")
            if not args.check:
                path.write_text(filled, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("\ndocumentation tables are stale; run scripts/render_docs.py")
        return 1
    print("documentation tables are up to date" if not stale else "\ndocs rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
