# SHL Conversational Assessment Recommender — Approach

**Public endpoint:** https://shl-recommender-ab5b.onrender.com
**Repo:** https://github.com/sudhir-yadav28/shl-
**Mean Recall@10 on the 10 public traces: 0.94**

## Design

The brief reads like a search problem but it isn't. A keyword search over the catalog already exists on SHL's site. The value here is in the *conversation*, and the rubric makes three things non-negotiable: schema compliance, catalog-only URLs, and recall against a domain-expert shortlist. Schema and grounding I can guarantee in code; recommendation quality is where the LLM helps.

So I avoided a "give the LLM tools and let it figure things out" design. Tool-use is great for multi-step reasoning, but here the actions are simple (clarify, recommend, refine, compare, refuse) and the cost of getting any one wrong shows up directly in the score. I went with a lightweight orchestrator: deterministic retrieval, one LLM call per turn for the language work, and a post-processor that enforces the schema.

## What happens on each `/chat` call

1. **Guardrail regex** catches obvious prompt-injection (`ignore previous instructions`, `act as`, etc.) and short-circuits.
2. **Embed** the latest user turn with Gemini `text-embedding-001` (3072-dim).
3. **Hybrid retrieval** — BM25 over name+description+keys plus dense cosine over the pre-computed embedding index, fused via Reciprocal Rank Fusion. Top 18.
4. **Canonical injection** — append OPQ32r, Verify G+, GSA and a few others to the candidate list so the LLM can pick them even when their description doesn't lexically match the query.
5. **One LLM call** classifies the action, drafts the reply, and picks names from the candidate block. Output is JSON-mode so parsing is deterministic.
6. **Shortlist builder** assembles ten items in priority order: tech-keyword force-includes → trigger-matched canonicals → LLM picks → retrieval rank.
7. **Pydantic validator** drops anything whose URL isn't in the catalog.

## Why the shortlist builder matters

Most of the score came from this layer.

- **Recall@10 has no precision penalty.** Under-filling is pure score loss, and the LLM was returning 1–3 names on refine turns. Filling to 10 lifted recall +0.40.
- **Some canonical SHL products (OPQ32r, Verify G+, GSA) appear in many reference shortlists** but their descriptions don't match the user's specific phrasing. Conditional force-include based on query triggers lifted recall +0.18.
- **Tech-keyword force-include** — when the user literally types `docker` or `aws` or `excel`, the matching catalog product gets a guaranteed slot. Lifted recall +0.16, mostly on C9 where the retriever was returning 5 Java variants and crowding out AWS/Docker/SQL.

Each slot in the final shortlist has a defensible reason for being there. "The user named this technology", "this is the standard SHL personality measure for senior selection", or "this is what semantic + lexical retrieval ranked highest". Not "the model picked it".

## Stack and why

- **FastAPI + Pydantic v2.** Spec-required; Pydantic is the last gate before any response leaves the service.
- **Gemini 2.0 Flash** as primary (15 RPM / 1500 RPD free tier — comfortable headroom), **Groq `llama-3.1-8b-instant`** as transparent fallback on 429 or any error. Two separate quotas matters in practice.
- **Gemini `text-embedding-001`** computed offline once, shipped as a 4.5 MB `embeddings.npy`. Cold start is 3s instead of 7 minutes.
- **`rank_bm25`** for lexical, NumPy matmul over 377 vectors for dense. FAISS would be overkill at this scale.
- **Render free Docker tier.** The spec's 2-minute cold-start allowance is essentially written for it.

## Evaluation

Replay harness in `eval/replay.py` parses each `Cn.md` for user turns and labeled URLs, replays multi-turn against the endpoint, and computes Recall@10. Per-trace:

| Trace | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Recall@10 | 1.00 | 0.80 | 0.75 | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.94** |

Behavior probes (`tests/test_behavior_probes.py`) — 9 scripted conversations with binary assertions: schema shape, no-recommend-on-vague-turn-1, prompt-injection refusal, off-topic refusal, refine keeps prior items, end-of-conversation only on confirmation, test_type letter codes. All 9 pass.

## What didn't work

- **Pure LLM-driven selection without top-up** plateaued at 0.43 Mean Recall. The model under-filled and picked similar-sounding products (Enterprise Leadership Report 2.0 instead of OPQ Leadership Report). The priority-ordered builder is what recovered the gap.
- **`sentence-transformers` locally** — PyTorch is ~500 MB and would have blown Render's 512 MB RAM cap. Switched to Gemini embeddings.
- **Gemini 2.5 Flash** as primary — 10 RPM / 250 RPD got exhausted within a single eval run. Dropped to 2.0 Flash and added Groq fallback.

## Resilience

When both Gemini and Groq fail (rate-limit or outage), the agent falls through to a retrieval-only path. The shortlist still populates via tech-keyword + canonical + retrieval rank, so Recall@10 is preserved without LLM availability. A signal-term heuristic decides between clarify and recommend so the resilience path doesn't violate the no-recommend-on-vague-turn-1 probe.

## What I'd add with more time

- LLM-based query rewriting before retrieval (would help C7's multi-faceted bilingual-healthcare-admin query).
- MMR-style diversity in retrieval to drop the hand-curated tech-keyword layer.
- Simulated-user harness with an LLM playing the user, to test holdout-style queries the way SHL actually evaluates.

## AI tool usage disclosure

I used Claude as a coding assistant — pair-debugging retrieval gaps, diagnosing the rate-limit cascade, and proposing the shortlist-builder priority ordering after I described the C8 and C9 trace failures. The design (single-LLM-call orchestrator over tool-use, canonical injection, tech-keyword layer, resilience path), the prompt content, and the keyword lists are mine. I verified every change against the 10 public traces and the probe suite before committing.
