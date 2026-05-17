"""LLM + embedding client factories. One place to swap providers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from google.api_core import exceptions as gax_exceptions

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EMBEDDING_MODEL = "models/gemini-embedding-001"
GENERATION_MODEL = "models/gemini-2.5-flash"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOG_PATH = DATA_DIR / "shl_product_catalog.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDING_IDS_PATH = DATA_DIR / "embedding_ids.json"


@lru_cache(maxsize=1)
def _configure_gemini() -> None:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_API_KEY)


def embed_query(text: str) -> np.ndarray:
    """Embed a query string. Returns an L2-normalized float32 vector.

    Uses task_type=retrieval_query so the embedding lives in the same space
    as the retrieval_document embeddings produced for the catalog.
    """
    _configure_gemini()
    try:
        resp = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_query",
        )
    except gax_exceptions.GoogleAPIError:
        raise
    vec = np.asarray(resp["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@lru_cache(maxsize=1)
def gemini_model():
    _configure_gemini()
    return genai.GenerativeModel(GENERATION_MODEL)
