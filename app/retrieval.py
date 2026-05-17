"""Hybrid retrieval: BM25 (lexical) + dense embeddings (semantic) fused via RRF.

The dense index is loaded from data/embeddings.npy (pre-computed offline).
The BM25 index is rebuilt from catalog text at startup (cheap — 377 docs).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app.catalog import Catalog, Product

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class HybridIndex:
    catalog: Catalog
    bm25: BM25Okapi
    embeddings: np.ndarray  # shape: (n_products, dim), L2-normalized
    id_order: list[str]     # product entity_id in the row order of embeddings
    row_for_id: dict[str, int]

    @classmethod
    def build(cls, catalog: Catalog, emb_path: Path, ids_path: Path) -> "HybridIndex":
        # BM25
        tokenized = [_tokenize(p.search_text()) for p in catalog.products]
        bm25 = BM25Okapi(tokenized)
        # Dense
        embeddings = np.load(emb_path)
        ids = json.loads(Path(ids_path).read_text())
        row_for_id = {pid: i for i, pid in enumerate(ids)}
        return cls(
            catalog=catalog,
            bm25=bm25,
            embeddings=embeddings,
            id_order=ids,
            row_for_id=row_for_id,
        )

    def bm25_rank(self, query: str, top_k: int = 50) -> list[tuple[Product, float]]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)[:top_k]
        return [(self.catalog.products[i], float(scores[i])) for i in order if scores[i] > 0]

    def dense_rank(
        self, query_vec: np.ndarray, top_k: int = 50
    ) -> list[tuple[Product, float]]:
        """query_vec must already be L2-normalized (so dot == cosine)."""
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.embeddings @ query_vec
        order = np.argsort(-sims)[:top_k]
        out: list[tuple[Product, float]] = []
        for row in order:
            pid = self.id_order[row]
            product = self.catalog.by_id.get(pid)
            if product is not None:
                out.append((product, float(sims[row])))
        return out

    def hybrid(
        self,
        query: str,
        query_vec: np.ndarray | None,
        top_k: int = 30,
        rrf_k: int = 60,
    ) -> list[tuple[Product, float]]:
        """Reciprocal Rank Fusion of BM25 and dense lists.

        RRF score for an item = sum over lists of 1 / (rrf_k + rank).
        rrf_k=60 is the canonical default. Items appearing in only one list still score.
        """
        scores: dict[str, float] = {}

        bm25_hits = self.bm25_rank(query, top_k=top_k * 2)
        for rank, (p, _) in enumerate(bm25_hits):
            scores[p.entity_id] = scores.get(p.entity_id, 0.0) + 1.0 / (rrf_k + rank)

        if query_vec is not None:
            dense_hits = self.dense_rank(query_vec, top_k=top_k * 2)
            for rank, (p, _) in enumerate(dense_hits):
                scores[p.entity_id] = scores.get(p.entity_id, 0.0) + 1.0 / (rrf_k + rank)

        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.catalog.by_id[pid], s) for pid, s in ordered]


def apply_filters(
    products: list[tuple[Product, float]],
    *,
    job_levels: list[str] | None = None,
    test_types: list[str] | None = None,
    languages: list[str] | None = None,
    max_duration_minutes: int | None = None,
    adaptive: bool | None = None,
) -> list[tuple[Product, float]]:
    """Drop products that violate hard constraints. Empty filter = pass-through.

    Filters are inclusive: if a product's job_levels overlap any filter level,
    it passes. Same for test_types and languages.
    """
    out: list[tuple[Product, float]] = []
    for p, s in products:
        if job_levels:
            if not (set(p.job_levels) & set(job_levels)):
                continue
        if test_types:
            product_types = {t.strip() for t in p.test_type.split(",") if t.strip()}
            if not (product_types & set(test_types)):
                continue
        if languages:
            if not (set(p.languages) & set(languages)):
                continue
        if max_duration_minutes is not None:
            d = _parse_duration_minutes(p.duration)
            if d is not None and d > max_duration_minutes:
                continue
        if adaptive is not None:
            want = "yes" if adaptive else "no"
            if p.adaptive != want:
                continue
        out.append((p, s))
    return out


def _parse_duration_minutes(duration: str) -> int | None:
    if not duration:
        return None
    m = re.search(r"(\d+)", duration)
    return int(m.group(1)) if m else None
