"""Replay the 10 conversation traces against /chat and score Recall@10.

For each Cn.md trace:
  1. Extract the user turns (verbatim from the trace).
  2. Extract the labeled expected shortlist (URLs in the last assistant turn).
  3. Replay user turns multi-turn through POST /chat, building history as we go.
  4. Compare final recommendation URLs against the expected URLs -> Recall@10.

Usage:
    source .venv/bin/activate
    # In another terminal: uvicorn app.main:app --port 8765
    python eval/replay.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

API_BASE = "http://127.0.0.1:8765"
TRACES_DIR = Path(__file__).parent / "traces"

USER_TURN_RE = re.compile(r"\*\*User\*\*\s*\n\s*>\s*(.+?)(?=\n\n|\Z)", re.S)
URL_RE = re.compile(r"https?://www\.shl\.com/products/product-catalog/view/[a-z0-9\-]+/?")


def parse_trace(path: Path) -> tuple[list[str], set[str]]:
    """Return (user_turns_in_order, expected_url_set_from_final_assistant_turn)."""
    text = path.read_text(encoding="utf-8")

    user_turns: list[str] = []
    for m in re.finditer(r"\*\*User\*\*\s*\n\n?\s*>\s*(.+?)\n\n", text, re.S):
        msg = m.group(1).strip()
        msg = re.sub(r"\n>\s*", "\n", msg)
        user_turns.append(msg.strip())

    blocks = text.split("### Turn ")
    final_block = blocks[-1]
    urls = set(URL_RE.findall(final_block))

    return user_turns, urls


def recall_at_k(predicted_urls: list[str], expected_urls: set[str], k: int = 10) -> float:
    if not expected_urls:
        return 0.0
    top_k = set(predicted_urls[:k])
    hits = top_k & expected_urls
    return len(hits) / len(expected_urls)


def replay_trace(name: str, user_turns: list[str], expected: set[str], client: httpx.Client) -> dict:
    history: list[dict] = []
    last_response: dict | None = None
    print(f"\n=== {name}: {len(user_turns)} user turns ===")
    for i, u in enumerate(user_turns, 1):
        history.append({"role": "user", "content": u})
        try:
            r = client.post(f"{API_BASE}/chat", json={"messages": history}, timeout=30.0)
            r.raise_for_status()
            resp = r.json()
        except Exception as e:
            print(f"  turn {i} FAILED: {e}")
            return {"trace": name, "error": str(e), "recall@10": 0.0}

        n_recs = len(resp.get("recommendations") or [])
        reply_short = resp.get("reply", "")[:80].replace("\n", " ")
        print(f"  turn {i}: recs={n_recs}  reply={reply_short!r}  eoc={resp.get('end_of_conversation')}")
        history.append({"role": "assistant", "content": resp["reply"]})
        last_response = resp
        if resp.get("end_of_conversation"):
            break
        time.sleep(float(os.environ.get("REPLAY_SLEEP", "3.0")))

    pred_urls = [r["url"] for r in (last_response or {}).get("recommendations", [])]
    rec = recall_at_k(pred_urls, expected, 10)
    hits = set(pred_urls[:10]) & expected
    missing = expected - set(pred_urls[:10])
    print(f"  Recall@10 = {rec:.2f}  ({len(hits)}/{len(expected)})")
    if missing:
        print(f"  Missing: {sorted(missing)[:3]}{'...' if len(missing) > 3 else ''}")
    return {
        "trace": name,
        "n_turns": len(user_turns),
        "n_expected": len(expected),
        "n_predicted": len(pred_urls),
        "hits": len(hits),
        "recall@10": rec,
        "missing": sorted(missing),
        "predicted": pred_urls,
    }


def main() -> None:
    traces = sorted(TRACES_DIR.glob("C*.md"), key=lambda p: int(re.search(r"C(\d+)", p.stem).group(1)))
    results = []
    with httpx.Client() as client:
        try:
            client.get(f"{API_BASE}/health", timeout=5.0).raise_for_status()
        except Exception as e:
            print(f"server not reachable at {API_BASE}: {e}")
            sys.exit(1)

        for path in traces:
            user_turns, expected = parse_trace(path)
            results.append(replay_trace(path.stem, user_turns, expected, client))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        marker = "OK" if r.get("recall@10", 0) >= 0.7 else ".."
        print(f"  [{marker}] {r['trace']:6s} recall@10={r.get('recall@10', 0):.2f}  ({r.get('hits', 0)}/{r.get('n_expected', 0)})")
    mean = sum(r.get("recall@10", 0) for r in results) / len(results)
    print(f"\n  Mean Recall@10 = {mean:.3f}")

    Path("eval/last_run.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Wrote eval/last_run.json")


if __name__ == "__main__":
    main()
