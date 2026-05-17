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
from groq import Groq

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EMBEDDING_MODEL = "models/gemini-embedding-001"
# gemini-2.0-flash has 15 RPM / 1500 RPD on free tier (vs 2.5-flash's 10/250).
# Quality is more than enough for our structured-JSON classification.
GENERATION_MODEL = "models/gemini-2.0-flash"
# Groq fallback: separate quota, different vendor.
# llama-3.1-8b-instant chosen for high TPM (14k) needed under burst.
GROQ_MODEL = "llama-3.1-8b-instant"
# Primary provider can be swapped via env. Default gemini; set PRIMARY_LLM=groq
# to flip (useful when Gemini daily quota is exhausted).
PRIMARY_LLM = os.environ.get("PRIMARY_LLM", "gemini").lower()

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


@lru_cache(maxsize=1)
def groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)
