"""Canonical / evergreen SHL products that should always be considered.

These are the catalog items that appear in MANY hiring contexts but whose
descriptions don't always lexically match the user's specific query (e.g. a
sales-restructuring query won't hit OPQ32r via BM25, but OPQ32r is still
often the right personality measure). We inject them into the candidate list
after hybrid retrieval so the LLM can pick them when contextually appropriate.

Identified from the 10 reference conversation traces — these recur across
many disparate personas.
"""

from __future__ import annotations

from app.catalog import Catalog, Product

CORE_PRODUCT_NAMES = [
    "Occupational Personality Questionnaire OPQ32r",
    "Global Skills Assessment",
    "Global Skills Development Report",
    "OPQ Leadership Report",
    "OPQ Universal Competency Report 2.0",
    "SHL Verify Interactive G+",
    "Smart Interview Live Coding",
    "Verify - General Ability Screen",
]


def core_products(catalog: Catalog) -> list[Product]:
    """Return Product objects for the canonical list (skipping any missing)."""
    out: list[Product] = []
    for name in CORE_PRODUCT_NAMES:
        p = catalog.find_by_name(name)
        if p is not None and p not in out:
            out.append(p)
    return out


def merge_with_core(
    ranked: list[tuple[Product, float]],
    catalog: Catalog,
    *,
    cap: int,
) -> list[tuple[Product, float]]:
    """Take the ranked retrieval output, append any core products not already
    present (with a small synthetic score so they sort last), and cap.
    """
    have = {p.entity_id for p, _ in ranked}
    merged = list(ranked)
    for cp in core_products(catalog):
        if cp.entity_id not in have:
            merged.append((cp, 0.0))
            have.add(cp.entity_id)
    return merged[:cap]
