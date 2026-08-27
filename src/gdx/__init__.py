"""Provenance-grounded document extraction with a bounded verification loop.

The structural claim of this package: an extracted field value must carry
provenance back to the specific source tokens it was read from, and a field
whose provenance does not verify is *abstained on* rather than emitted.

Extraction therefore becomes **selection plus verification** rather than
generation. The consequence that motivates the whole repository is a guarantee
rather than a statistic: a value that does not appear anywhere in the document
cannot be emitted at all, because the only thing the model can output is an
index range into the document's own tokens.

See :mod:`gdx.models.heads` for the span-selection head, :mod:`gdx.verify` for
the bounded verification loop, and ``docs/METHOD.md`` for the derivations.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
