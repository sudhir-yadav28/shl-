"""On-disk JSON cache keyed on prompt hash.

Enabled when env var SHL_LLM_CACHE_DIR is set. The production deployment does
NOT set it, so it has no effect there. Local dev/eval reads/writes
eval/llm_cache/<sha>.json so re-running the harness doesn't re-burn quota.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _cache_dir() -> Path | None:
    d = os.environ.get("SHL_LLM_CACHE_DIR")
    if not d:
        return None
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


def get(prompt: str) -> dict | None:
    d = _cache_dir()
    if d is None:
        return None
    f = d / f"{_key(prompt)}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return None
    return None


def put(prompt: str, value: dict) -> None:
    d = _cache_dir()
    if d is None:
        return
    f = d / f"{_key(prompt)}.json"
    f.write_text(json.dumps(value))
