"""Print how to obtain a real document dataset. Downloads nothing.

    python scripts/download_real.py --dataset funsd

This repository's default path is offline: no keys, no downloads. Real datasets
are supported by an *optional* loader (:mod:`gdx.data.real`) and **no committed
number uses one**, because none of them records the token span each annotated
value was read from -- which is the oracle this project's grounding metric needs.
The script prints the source, the licence and the expected on-disk layout, and
leaves the decision to the reader.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from gdx.data.real import SUPPORTED

INFO = {
    "funsd": {
        "name": "FUNSD (Form Understanding in Noisy Scanned Documents)",
        "paper": "Jaume, Ekenel & Thiran, 2019",
        "url": "https://guillaumejaume.github.io/FUNSD/",
        "licence": "research use; see the dataset page",
        "note": "Annotates generic question/answer links, not invoice fields.",
    },
    "cord": {
        "name": "CORD (Consolidated Receipt Dataset)",
        "paper": "Park et al., 2019",
        "url": "https://github.com/clovaai/cord",
        "licence": "CC BY 4.0",
        "note": "Receipts with total/subtotal/tax annotations; closest to this schema.",
    },
    "sroie": {
        "name": "SROIE (ICDAR 2019 Scanned Receipts OCR and IE)",
        "paper": "Huang et al., 2019",
        "url": "https://rrc.cvc.uab.es/?ch=13",
        "licence": "registration required",
        "note": "Four fields: company, date, address, total.",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(SUPPORTED), default=None)
    parser.add_argument("--root", default="data/raw")
    args = parser.parse_args(argv)

    names = [args.dataset] if args.dataset else list(SUPPORTED)
    for name in names:
        info = INFO[name]
        print(f"\n=== {name} ===")
        print(f"  {info['name']} ({info['paper']})")
        print(f"  source:  {info['url']}")
        print(f"  licence: {info['licence']}")
        print(f"  note:    {info['note']}")
        print(f"  expected layout: {args.root}/{name}/test.jsonl")
        print("    one JSON object per line: {id, width, height, tokens:[{text,box,page}], fields}")
    print(
        "\nNothing was downloaded. gdx.data.real.load_real_dataset reads the layout above and\n"
        "recovers provenance spans by string matching, which is an approximation -- see the\n"
        "module docstring and docs/RESULTS.md for why no shipped number uses this path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
