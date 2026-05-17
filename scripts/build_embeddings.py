"""One-time script: embed every catalog product with Gemini text-embedding-004
and save the matrix to data/embeddings.npy plus a parallel id list.

Run locally once, then commit the outputs. Cold start on Render then just
np.load() instead of re-embedding 377 docs over the network.

Usage:
    source .venv/bin/activate
    python scripts/build_embeddings.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import load_catalog

load_dotenv()

import google.generativeai as genai
from google.api_core import exceptions as gax_exceptions

API_KEY = os.environ["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

EMBEDDING_MODEL = "models/gemini-embedding-001"
PROGRESS_EVERY = 50
# Free tier: 100 req/min. 700ms inter-request -> ~85 req/min, comfortable margin.
SLEEP_BETWEEN = 0.7

CATALOG_PATH = Path("data/shl_product_catalog.json")
EMB_OUT = Path("data/embeddings.npy")
IDS_OUT = Path("data/embedding_ids.json")


def main() -> None:
    catalog = load_catalog(CATALOG_PATH)
    print(f"Loaded {len(catalog.products)} products")

    texts = [p.search_text() for p in catalog.products]
    ids = [p.entity_id for p in catalog.products]

    def embed_one(text: str) -> list[float]:
        for attempt in range(5):
            try:
                resp = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document",
                )
                return resp["embedding"]
            except gax_exceptions.ResourceExhausted as e:
                wait = 30 + 10 * attempt
                print(f"    rate-limited, sleeping {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
        raise RuntimeError(f"giving up on text: {text[:80]}")

    all_vectors: list[list[float]] = []
    t0 = time.time()
    for idx, text in enumerate(texts, start=1):
        all_vectors.append(embed_one(text))
        if idx % PROGRESS_EVERY == 0 or idx == len(texts):
            print(f"  embedded {idx}/{len(texts)}  ({time.time() - t0:.0f}s)")
        if idx < len(texts):
            time.sleep(SLEEP_BETWEEN)
    elapsed = time.time() - t0

    matrix = np.asarray(all_vectors, dtype=np.float32)
    assert matrix.shape[0] == len(texts), f"unexpected row count {matrix.shape}"
    print(f"  embedding dim: {matrix.shape[1]}")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix_normed = matrix / norms

    EMB_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_OUT, matrix_normed)
    IDS_OUT.write_text(json.dumps(ids))

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  wrote {EMB_OUT}  shape={matrix_normed.shape}  size={EMB_OUT.stat().st_size / 1024:.0f} KB")
    print(f"  wrote {IDS_OUT}")


if __name__ == "__main__":
    main()
