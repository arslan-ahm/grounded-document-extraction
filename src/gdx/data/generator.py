"""Procedural generator of synthetic visually-rich invoices.

Ground truth is *by construction*. The generator places every token, so it knows
the exact token index range each field's value was written at, the exact box that
range covers, and whether that written form literally equals the target the
extractor is scored against. No annotated real dataset provides the first two,
which is why the shipped results come from here.

The difficulty is not incidental. Each of the following exists because it breaks
a specific class of extractor, and each is a config switch so its contribution
can be measured rather than assumed:

* **Distractor amounts.** "Subtotal", "Balance Forward" and "Amount Paid" sit
  immediately above "Total Due". A bottom-right-most-number heuristic picks one.
* **Repeated values.** "Amount Due" reprints the total under a label that is
  *also* a synonym for total; the invoice id reprints on every continuation
  page. Both are recorded in ``alt_spans`` so grounding is scored fairly.
* **Missing fields.** A dropped field makes *abstention* the correct output, so
  the abstention machinery is measured against cases where emitting anything is
  an error.
* **Multi-page.** The summary block lands on the last page, far from the header,
  so the model cannot rely on absolute position.
* **Box noise and page skew.** Row membership stops being an equality on ``y``.
* **Non-ISO dates.** With probability ``verbose_date_prob`` a date is written in
  a form no contiguous span equals -- the structural limitation of selection.

Amount arithmetic is done in integer cents so ``subtotal + tax == total`` is
exact, which is what lets the verification loop treat a mismatch as a real
inconsistency rather than a floating-point artefact.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np

from gdx.config import DataConfig
from gdx.data import lexicon as lex
from gdx.data.layout import MARGIN, LayoutBuilder
from gdx.data.schema import FIELDS, Document, FieldTruth, normalise_text

DROPPABLE = ("po_number", "due_date", "subtotal", "tax")
#: Extra drop weight per droppable field. A purchase-order reference is missing
#: far more often than a subtotal in real invoice populations.
DROP_WEIGHT = {"po_number": 2.2, "due_date": 0.8, "subtotal": 0.5, "tax": 0.6}
TAX_RATES = (0.05, 0.10, 0.20)


def _money(cents: int, style: dict[str, str]) -> list[str]:
    """Render integer cents in a document's currency style as word tokens."""
    whole, frac = divmod(abs(int(cents)), 100)
    digits = f"{whole:,}" if style["thousands"] == "1" else str(whole)
    body = f"{digits}.{frac:02d}"
    sign = "-" if cents < 0 else ""
    sym = style["symbol"]
    if not sym:
        return [f"{sign}{body}"]
    if style["attached"] == "1":
        return [f"{sym}{sign}{body}"]
    return [sym, f"{sign}{body}"]


def _date_forms(day: _dt.date, rng: np.random.Generator, non_iso_prob: float) -> list[str]:
    """Words for a written date; ISO with probability ``1 - non_iso_prob``."""
    if rng.random() >= non_iso_prob:
        return [day.isoformat()]
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return [f"{day.day:02d}/{day.month:02d}/{day.year}"]
    if kind == 1:
        return [lex.MONTH_ABBR[day.month - 1], f"{day.day},", str(day.year)]
    return [lex.ordinal(day.day), "of", lex.MONTH_FULL[day.month - 1], str(day.year)]


def _sample_content(rng: np.random.Generator, cfg: DataConfig, n_items: int) -> dict:
    """Draw the semantic content of one invoice, with exact cent arithmetic."""
    style = lex.CURRENCY_STYLES[int(rng.integers(0, len(lex.CURRENCY_STYLES)))]
    vendor = [
        lex.VENDOR_FIRST[int(rng.integers(0, len(lex.VENDOR_FIRST)))],
        lex.VENDOR_SECOND[int(rng.integers(0, len(lex.VENDOR_SECOND)))],
    ]
    if rng.random() < 0.6:
        vendor.append(lex.VENDOR_SUFFIX[int(rng.integers(0, len(lex.VENDOR_SUFFIX)))])
    inv_prefix = lex.ID_PREFIXES[int(rng.integers(0, len(lex.ID_PREFIXES)))]
    inv_id = f"{inv_prefix}-{int(rng.integers(100, 99_999)):05d}"
    po_prefix = lex.PO_PREFIXES[int(rng.integers(0, len(lex.PO_PREFIXES)))]
    po_id = f"{po_prefix}-{int(rng.integers(100, 99_999)):05d}"
    issued = _dt.date(2023, 1, 1) + _dt.timedelta(days=int(rng.integers(0, 900)))
    due = issued + _dt.timedelta(days=int(rng.choice([14, 21, 30, 45, 60])))

    items = []
    for _ in range(n_items):
        desc = [lex.ITEM_QUALIFIER[int(rng.integers(0, len(lex.ITEM_QUALIFIER)))],
                lex.ITEM_WORDS[int(rng.integers(0, len(lex.ITEM_WORDS)))]]
        if rng.random() < 0.45:
            desc.append(lex.ITEM_SIZE[int(rng.integers(0, len(lex.ITEM_SIZE)))])
        qty = int(rng.integers(1, 25))
        unit = int(rng.integers(150, 45_000))
        items.append({"desc": desc, "qty": qty, "unit": unit, "amount": qty * unit})

    subtotal = sum(i["amount"] for i in items)
    rate = float(rng.choice(TAX_RATES))
    tax = int(round(subtotal * rate))
    return {
        "style": style,
        "vendor": vendor,
        "invoice_id": inv_id,
        "po_number": po_id,
        "issued": issued,
        "due": due,
        "items": items,
        "subtotal": subtotal,
        "tax": tax,
        "tax_rate": rate,
        "total": subtotal + tax,
        "table_headers": lex.TABLE_HEADERS[int(rng.integers(0, len(lex.TABLE_HEADERS)))],
        "labels": {
            name: syns[int(rng.integers(0, len(syns)))]
            for name, syns in lex.LABEL_SYNONYMS.items()
        },
        "duplicate_total": bool(rng.random() < cfg.duplicate_total_prob),
        "n_distractors": int(rng.integers(1, 4)) if rng.random() < cfg.distractor_prob else 0,
    }


def _choose_dropped(rng: np.random.Generator, cfg: DataConfig) -> set[str]:
    """Which fields are absent from this document (abstention is then correct)."""
    dropped = set()
    for name in DROPPABLE:
        p = min(0.95, cfg.missing_field_prob * DROP_WEIGHT[name])
        if rng.random() < p:
            dropped.add(name)
    return dropped


def _budget_items(cfg: DataConfig, multi_page: bool) -> int:
    """Largest line-item count that keeps the document inside ``max_tokens``.

    Overheads are measured constants from the layout below, not guesses; the
    generator asserts the final count afterwards and
    ``test_generated_documents_fit_the_token_budget`` enforces it.
    """
    overhead = 66 + (14 if multi_page else 0)
    per_row = 8
    room = (cfg.max_tokens - overhead) // per_row
    return int(max(2, min(9, room)))


def _write_header(lb: LayoutBuilder, c: dict, dropped: set[str], cfg: DataConfig,
                  rng: np.random.Generator, truths: dict[str, FieldTruth]) -> None:
    """Vendor block, title, and the label/value meta grid."""
    lb.write(["INVOICE"], 0.60, 0.045, size=1.9)
    y = 0.05
    if rng.random() < 0.5:
        lb.write([c["labels"]["vendor_name"] + ":"], MARGIN, y, size=0.85)
        y += 0.024
    span = lb.write(c["vendor"], MARGIN, y, size=1.25)
    truths["vendor_name"] = FieldTruth(
        name="vendor_name", value=" ".join(c["vendor"]), present=True,
        span=span, span_text=" ".join(c["vendor"]), requires_normalisation=False,
    )
    y += 0.036
    street = lex.STREETS[int(rng.integers(0, len(lex.STREETS)))]
    city = lex.CITIES[int(rng.integers(0, len(lex.CITIES)))]
    lb.write([str(int(rng.integers(1, 200)))] + street.split(), MARGIN, y, size=0.8)
    lb.write([city], MARGIN, y + 0.022, size=0.8)

    grid_y = 0.15
    order = ("invoice_id", "invoice_date", "due_date", "po_number")
    for name in order:
        if name in dropped:
            truths[name] = FieldTruth(name=name, value="", present=False)
            continue
        if name == "invoice_id":
            words, target = [c["invoice_id"]], c["invoice_id"]
        elif name == "po_number":
            words, target = [c["po_number"]], c["po_number"]
        else:
            day = c["issued"] if name == "invoice_date" else c["due"]
            words = _date_forms(day, rng, cfg.verbose_date_prob)
            target = day.isoformat()
        lb.write(c["labels"][name].split(), 0.58, grid_y, size=0.85)
        span = lb.write_right_aligned(words, 0.94, grid_y, size=0.85)
        written = " ".join(words)
        truths[name] = FieldTruth(
            name=name, value=target, present=True, span=span, span_text=written,
            requires_normalisation=normalise_text(written) != normalise_text(target),
        )
        grid_y += 0.030


def _write_table(
    lb: LayoutBuilder, c: dict, n_pages: int
) -> tuple[list[tuple[int, int]], float]:
    """The line-item table, spilling across pages.

    Returns:
        The token span of each line-item amount, and the ``y`` the summary block
        should start below.
    """
    per_page = -(-len(c["items"]) // n_pages)
    amount_spans: list[tuple[int, int]] = []
    y = 0.31
    idx = 0
    for page in range(n_pages):
        if page > 0:
            lb.new_page()
            lb.write(["Invoice", "No", c["invoice_id"]], MARGIN, 0.05, size=0.85)
            y = 0.12
        lb.write([c["table_headers"][0]], MARGIN, y, size=0.85)
        lb.write([c["table_headers"][1]], 0.55, y, size=0.85)
        lb.write([c["table_headers"][2]], 0.66, y, size=0.85)
        lb.write_right_aligned([c["table_headers"][3]], 0.94, y, size=0.85)
        y += 0.030
        for item in c["items"][idx : idx + per_page]:
            lb.write(item["desc"], MARGIN, y, size=0.85)
            lb.write([str(item["qty"])], 0.55, y, size=0.85)
            lb.write(_money(item["unit"], c["style"]), 0.66, y, size=0.85)
            amount_spans.append(
                lb.write_right_aligned(_money(item["amount"], c["style"]), 0.94, y, size=0.85)
            )
            y += 0.028
        idx += per_page
        lb.write(["Page", str(page + 1), "of", str(n_pages)], MARGIN, 0.955, size=0.75)
    return amount_spans, y


def _write_summary(lb: LayoutBuilder, c: dict, dropped: set[str], y: float,
                   rng: np.random.Generator, truths: dict[str, FieldTruth]) -> None:
    """Distractor lines, then the summary amounts, then the optional duplicate.

    The distractors come *first* so the target amounts are not simply the last
    numbers on the page; a positional shortcut has to be defeated for the
    benchmark to measure reading rather than counting from the bottom.
    """
    used = {c["subtotal"], c["tax"], c["total"]}
    for k in range(c["n_distractors"]):
        label = lex.DISTRACTOR_LABELS[int(rng.integers(0, len(lex.DISTRACTOR_LABELS)))]
        cents = int(rng.integers(500, max(2000, c["total"])))
        while cents in used:
            cents += 7
        used.add(cents)
        lb.write(label.split(), 0.58, y, size=0.85)
        lb.write_right_aligned(_money(cents, c["style"]), 0.94, y, size=0.85)
        y += 0.028
        del k

    for name, cents in (("subtotal", c["subtotal"]), ("tax", c["tax"]), ("total", c["total"])):
        if name in dropped:
            truths[name] = FieldTruth(name=name, value="", present=False)
            continue
        label = c["labels"][name].split()
        if name == "tax":
            label = label + [f"({c['tax_rate'] * 100:.0f}%)"]
        lb.write(label, 0.58, y, size=0.9 if name != "total" else 1.05)
        words = _money(cents, c["style"])
        span = lb.write_right_aligned(words, 0.94, y, size=0.9 if name != "total" else 1.05)
        written = " ".join(words)
        truths[name] = FieldTruth(
            name=name, value=written, present=True, span=span, span_text=written,
            requires_normalisation=False,
        )
        y += 0.030

    if c["duplicate_total"] and "total" not in dropped:
        lb.write(["Amount", "Due"], 0.58, y, size=0.85)
        lb.write_right_aligned(_money(c["total"], c["style"]), 0.94, y, size=0.85)
        y += 0.028
    line = lex.FOOTER_LINES[int(rng.integers(0, len(lex.FOOTER_LINES)))]
    lb.write(list(line), MARGIN, min(0.93, y + 0.02), size=0.75)


def _collect_alt_spans(doc: Document, max_span_len: int = 5) -> None:
    """Record every *other* span whose canonical value equals a field's target.

    Done by scanning the finished document rather than by bookkeeping during
    layout, which catches the coincidences a bookkeeping approach would miss --
    a line-item amount that happens to equal the tax, an "Amount Due" duplicate,
    the invoice id reprinted on a continuation page. Grounding is then scored
    against the set of *all* correct locations, so a model that reads the value
    from the duplicate is not marked wrong for it.
    """
    from gdx.data.schema import canonical_value

    n = len(doc.tokens)
    for name, truth in doc.fields.items():
        if not truth.present or truth.span is None:
            continue
        target = canonical_value(name, truth.value)
        if not target:
            continue
        alts: list[tuple[int, int]] = []
        for start in range(n):
            page = doc.tokens[start].page
            for end in range(start, min(start + max_span_len, n)):
                if doc.tokens[end].page != page:
                    break
                if (start, end) == truth.span:
                    continue
                if canonical_value(name, doc.span_text(start, end)) == target:
                    alts.append((start, end))
        truth.alt_spans = alts


def generate_document(doc_id: int, cfg: DataConfig, seed: int | None = None) -> Document:
    """Generate one invoice with exact provenance ground truth.

    Args:
        doc_id: Identifier, also mixed into the RNG so a document is a pure
            function of ``(doc_id, seed, cfg)``.
        cfg: Generator configuration.
        seed: Base seed. Defaults to ``cfg.seed``.

    Returns:
        A :class:`~gdx.data.schema.Document`.

    Raises:
        ValueError: If ``cfg.max_tokens`` is too small to hold a two-item invoice
            even after shrinking the table. Silently truncating instead would
            delete the summary block and with it the ground truth, so this is a
            hard error.
    """
    base = cfg.seed if seed is None else seed
    n_items = None
    for attempt in range(7):
        rng = np.random.default_rng((base * 1_000_003 + doc_id * 7919 + attempt) % (2**63 - 1))
        multi = bool(rng.random() < cfg.multi_page_prob) and cfg.max_pages > 1
        if n_items is None:
            n_items = _budget_items(cfg, multi)
        n_pages = 1
        if multi:
            n_pages = 2 if (rng.random() < 0.7 or cfg.max_pages < 3) else 3
            n_pages = min(n_pages, cfg.max_pages, max(1, n_items))
        content = _sample_content(rng, cfg, n_items)
        dropped = _choose_dropped(rng, cfg)
        truths: dict[str, FieldTruth] = {}
        lb = LayoutBuilder()
        _write_header(lb, content, dropped, cfg, rng, truths)
        item_spans, y = _write_table(lb, content, n_pages)
        _write_summary(lb, content, dropped, max(y + 0.03, 0.62), rng, truths)
        if len(lb.tokens) <= cfg.max_tokens:
            lb.apply_noise(cfg.box_jitter, cfg.rotation_deg, rng)
            doc = Document(
                doc_id=doc_id,
                tokens=lb.tokens,
                fields={name: truths[name] for name in FIELDS},
                n_pages=lb.n_pages,
                meta={
                    "n_items": n_items,
                    "n_distractors": content["n_distractors"],
                    "duplicate_total": content["duplicate_total"],
                    "tax_rate": content["tax_rate"],
                    "subtotal_cents": content["subtotal"],
                    "tax_cents": content["tax"],
                    "total_cents": content["total"],
                    "dropped": sorted(dropped),
                    "line_item_spans": item_spans,
                    "currency_symbol": content["style"]["symbol"],
                    "n_tokens": len(lb.tokens),
                },
            )
            _collect_alt_spans(doc)
            return doc
        n_items = max(1, n_items - 1)
    raise ValueError(
        f"cannot fit a document in max_tokens={cfg.max_tokens}; raise it to at least 128"
    )


def generate_dataset(
    n: int, cfg: DataConfig, seed: int | None = None, id_offset: int = 0
) -> list[Document]:
    """Generate ``n`` documents. Deterministic in ``(seed, id_offset, cfg)``."""
    return [generate_document(id_offset + i, cfg, seed) for i in range(n)]
