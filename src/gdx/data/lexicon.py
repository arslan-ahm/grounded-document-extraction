"""Surface vocabulary for the document generator.

Kept in one module so the *lexical* difficulty of the benchmark is inspectable
and adjustable independently of the layout. Two properties matter:

* **Label synonyms are plural.** ``total`` is written as "Total", "Total Due",
  "Amount Due", "Balance Due" or "TOTAL". A heuristic baseline keyed on one
  string would be a strawman, so the synonym lists are shared with
  :mod:`gdx.baselines.heuristic` and the baseline is allowed to use all of them.
* **Distractor labels look like target labels.** "Subtotal", "Balance Forward"
  and "Amount Paid" sit next to "Total Due" carrying different amounts, which is
  the failure mode that makes invoice extraction non-trivial in practice.
"""

from __future__ import annotations

VENDOR_FIRST = (
    "Northwind", "Aldergate", "Bluecrest", "Corvus", "Delphi", "Eastbrook",
    "Fairmount", "Granite", "Halcyon", "Ironvale", "Juniper", "Kestrel",
    "Larchmont", "Meridian", "Norwood", "Orchard", "Pinnacle", "Quarry",
    "Redgate", "Silverpine", "Thornbury", "Umberton", "Vantage", "Westfield",
)

VENDOR_SECOND = (
    "Industrial", "Logistics", "Supply", "Systems", "Technical", "Trading",
    "Fabrication", "Analytics", "Instruments", "Components", "Materials",
)

VENDOR_SUFFIX = ("Ltd", "LLC", "GmbH", "Inc", "Co", "PLC", "SA")

ITEM_WORDS = (
    "Bracket", "Coupling", "Filter", "Gasket", "Housing", "Impeller", "Journal",
    "Bearing", "Bushing", "Clamp", "Manifold", "Nozzle", "Pinion", "Retainer",
    "Seal", "Spacer", "Sleeve", "Valve", "Washer", "Adapter", "Cartridge",
    "Diaphragm", "Element", "Flange", "Grommet",
)

ITEM_QUALIFIER = (
    "Steel", "Brass", "Nylon", "Alloy", "Ceramic", "Composite", "Titanium",
    "Bronze", "Graphite", "Polymer",
)

ITEM_SIZE = ("M6", "M8", "M10", "M12", "1/4in", "3/8in", "1/2in", "DN25", "DN40")

STREETS = ("Mill Road", "Harbour Way", "Kiln Street", "Foundry Lane", "Quay Road")
CITIES = ("Bristol", "Leeds", "Utrecht", "Aarhus", "Lyon", "Porto", "Gdansk")

#: Label synonyms per field. Order is significant only in that the generator
#: samples uniformly from the list; the heuristic baseline matches against all.
LABEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "invoice_id": ("Invoice No", "Invoice Number", "Invoice #", "Inv No", "Document No"),
    "invoice_date": ("Invoice Date", "Date", "Date Issued", "Issued"),
    "due_date": ("Due Date", "Payment Due", "Due By", "Terms Due"),
    "vendor_name": ("From", "Supplier", "Vendor", "Remit To"),
    "po_number": ("PO Number", "Purchase Order", "PO Ref", "Customer PO"),
    "subtotal": ("Subtotal", "Sub Total", "Net Amount", "Goods Total"),
    "tax": ("Tax", "VAT", "Sales Tax", "Tax Amount"),
    "total": ("Total", "Total Due", "Amount Due", "Balance Due", "TOTAL"),
}

#: Labels that carry an amount but are *not* an extraction target. These are the
#: distractors: a heuristic that grabs the nearest number to the bottom-right of
#: the page will pick one of these.
DISTRACTOR_LABELS: tuple[str, ...] = (
    "Balance Forward",
    "Amount Paid",
    "Previous Balance",
    "Shipping",
    "Discount",
    "Deposit Held",
)

#: Column headers for the line-item table.
TABLE_HEADERS: tuple[tuple[str, ...], ...] = (
    ("Description", "Qty", "Unit", "Amount"),
    ("Item", "Quantity", "Rate", "Total"),
    ("Details", "Units", "Price", "Value"),
)

FOOTER_LINES = (
    ("Thank", "you", "for", "your", "business"),
    ("Payment", "within", "30", "days", "of", "invoice", "date"),
    ("Registered", "office", "as", "above"),
)

#: Currency rendering styles. Each document commits to exactly one, so a model
#: cannot rely on a global convention.
CURRENCY_STYLES: tuple[dict[str, str], ...] = (
    {"symbol": "$", "attached": "1", "thousands": "1"},
    {"symbol": "$", "attached": "0", "thousands": "0"},
    {"symbol": "EUR", "attached": "0", "thousands": "1"},
    {"symbol": "", "attached": "1", "thousands": "1"},
    {"symbol": "GBP", "attached": "0", "thousands": "0"},
)

ID_PREFIXES = ("INV", "IN", "BILL", "DOC")
PO_PREFIXES = ("PO", "ORD")

MONTH_ABBR = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

MONTH_FULL = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def ordinal(day: int) -> str:
    """English ordinal suffix, e.g. ``3 -> "3rd"``."""
    if 10 <= day % 100 <= 20:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }".replace(" ", "")


def build_vocab(extra: tuple[str, ...] = ()) -> list[str]:
    """The closed vocabulary the token embedder is built over.

    Anything outside it is hashed into a bucket, which is how numeric literals
    -- unbounded by construction -- are handled. Returned sorted so the mapping
    is stable across processes regardless of set iteration order.
    """
    words: set[str] = set()
    for group in (
        VENDOR_FIRST, VENDOR_SECOND, VENDOR_SUFFIX, ITEM_WORDS, ITEM_QUALIFIER,
        ITEM_SIZE, STREETS, CITIES, DISTRACTOR_LABELS, MONTH_ABBR, MONTH_FULL,
        ID_PREFIXES, PO_PREFIXES, extra,
    ):
        for phrase in group:
            words.update(phrase.split())
    for syns in LABEL_SYNONYMS.values():
        for phrase in syns:
            words.update(phrase.split())
    for headers in TABLE_HEADERS:
        words.update(headers)
    for line in FOOTER_LINES:
        words.update(line)
    words.update({"INVOICE", "Page", "of", "x", "@", ":", "#", "%"})
    return sorted(words)
