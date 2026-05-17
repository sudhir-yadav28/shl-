# SHL Conversational Assessment Recommender — Approach

**Public endpoint:** `https://shl-recommender-ab5b.onrender.com`
(`GET /health` → `{"status":"ok"}`, `POST /chat` per spec)

**Repo:** `https://github.com/sudhir-yadav28/shl-`

## Design

A stateless FastAPI service with a single-LLM-call orchestrator per turn:

```
POST /chat
  → guardrail regex (injection refusal)
  → embed user query (Gemini text-embedding-001)
  → hybrid retrieval: BM25 + dense, fused via Reciprocal Rank Fusion
  → canonical-product injection (OPQ32r, GSA, Verify G+, etc. always in candidates)
  → 1 LLM call: classifies action ∈ {clarify, recommend, refine, compare, refuse}
                + drafts reply
                + selects shortlist by exact name from candidates list
  → shortlist builder (priority order):
      a) tech-keyword force-include   (e.g. "docker"   → Docker (New))
      b) trigger-matched canonicals   (OPQ32r unconditional + others conditional)
      c) LLM-picked items
      d) retrieval-ranked top-up to 10
  → Pydantic schema validator (URL must exist in catalog, ≤10 items)
```

I chose this over a pure-LLM-tool-use design because the rubric rewards
schema compliance, catalog-grounded URLs, and Mean Recall@10 — all three
are easier to guarantee in code than to prompt the model into.

## Retrieval

- **Corpus:** 377 Individual Test Solutions parsed from
  `shl_product_catalog.json`. `test_type` letter codes (K, P, A, etc.) are
  derived from the `keys` field per the convention used in the sample traces.
- **Lexical:** `rank_bm25` over `name + description + keys + job_levels`.
- **Dense:** 3072-dim vectors from `gemini-embedding-001` (`task_type=retrieval_document`),
  pre-computed offline and shipped as a 4.5 MB `embeddings.npy`. Cold start on
  Render takes ~3s (load + index build) instead of ~7 min (re-embed via API).
- **Fusion:** Reciprocal Rank Fusion (k=60). Top-18 retrieval + ~8 canonical
  products = ~25 candidates supplied to the LLM.

## Prompt design

System prompt enumerates the 4 behaviors (clarify / recommend / refine / compare)
plus refusal rules. The per-turn prompt embeds the conversation history and the
candidate block (`[i] name | meta | description`). LLM returns JSON:
`{action, reply, recommendation_names[], end_of_conversation}`. Output is in
JSON mode (`response_mime_type=application/json` on Gemini, `response_format` on
Groq) to keep parses deterministic.

**Shortlist top-up.** Early experiments showed the LLM picking only 1–3 items
on multi-turn refines, leaving 7+ slots empty. Since Recall@10 has no precision
penalty, I always fill to 10: (a) trigger-keyword canonicals, (b) LLM picks,
(c) retrieval rank. This single change lifted mean recall from 0.43 → 0.83.

**Tech-keyword force-include.** When a user names a technology literally
("docker", "aws", "excel", "hipaa"), the agent force-includes the exact catalog
product even if it would have ranked low. Lifted mean recall 0.83 → 0.94.
Hand-curated from the catalog — small (~20 entries), defensible.

**Refine state recovery.** The API is stateless, so on a refine turn the agent
parses the previous assistant `content` for product names (case-insensitive
substring against the catalog name set). If the LLM otherwise returned empty
names, we fall back to the recovered shortlist instead of dropping to clarify.

## Stack & deployment

| Layer | Choice | Why |
|---|---|---|
| Web | FastAPI + Uvicorn + Pydantic v2 | Spec-required; strict schema enforcement at the boundary |
| LLM primary | Gemini 2.0 Flash | 15 RPM / 1500 RPD free tier — comfortable headroom |
| LLM fallback | Groq llama-3.1-8b-instant | Separate vendor / quota; transparent failover on 429 |
| Embeddings | Gemini text-embedding-001 (3072-dim) | Free; pre-computed offline → 4.5 MB shipped in repo |
| Lexical | `rank_bm25` | Tiny dep, no infra |
| Dense | NumPy matmul over 377 vectors | FAISS would be overkill at this scale |
| Deploy | Render free tier (Docker) | Spec's 2-min cold-start allowance fits free tier exactly |

## Guardrails

- **Pre-LLM regex** for prompt-injection patterns (`ignore previous instructions`,
  `act as`, `system prompt`, etc.) — short-circuit with refusal.
- **Off-topic & off-scope** handled by the LLM's system prompt (catalog-only).
- **URL whitelist** — every recommendation URL is validated against the catalog
  before responding. Hallucinated names are dropped silently.
- **Schema enforcement** — Pydantic models on input and output. Off-schema
  responses cause 5xx, never silently malformed JSON.

## Evaluation

Local replay harness (`eval/replay.py`) parses each `Cn.md` for user turns
and labeled URLs, replays multi-turn against the live endpoint, and computes
Recall@10 per trace.

**Mean Recall@10 on the 10 public traces: 0.935.**
Per-trace: C1=1.00 C2=0.80 C3=0.75 C4=1.00 C5=0.80 C6=1.00 C7=1.00 C8=1.00 C9=1.00 C10=1.00.

Behavior probes (`tests/test_behavior_probes.py`) cover: schema shape, no
recommend on vague turn 1, refusal of off-topic and prompt-injection, catalog-only
URLs, refine keeps prior items, end_of_conversation only on confirmation,
test_type is a valid letter code.

## What didn't work

- **Pure LLM-driven selection without top-up** plateaued around 0.43 Mean Recall.
  The LLM consistently under-filled the shortlist (1–3 items) and picked
  similar-but-different items (e.g. Enterprise Leadership Report 2.0 in place
  of OPQ Leadership Report). The shortlist-builder priority queue is what
  recovered most of the gap.
- **Sentence-transformers locally** (initial plan): would have added 500 MB
  of PyTorch deps, breaching Render's 512 MB RAM cap. Switched to Gemini
  embeddings + on-disk pre-compute.
- **Gemini 2.5 Flash** as primary: 10 RPM / 250 RPD free-tier quota
  exhausted within a single eval run. Dropped to gemini-2.0-flash (15 / 1500)
  and added Groq fallback.

## AI-tool usage disclosure

I used Claude Code as a coding assistant: pair-debugging the retrieval gaps,
diagnosing the rate-limit cascade, suggesting the top-up + force-include
shortlist strategy. All design decisions, prompt content, keyword lists,
and per-component tradeoffs were mine — the assistant proposes and explains;
I drive the architecture and verify each change against the 10 traces and
behavior probes.

## What I'd add with more time

- **LLM-based query rewriting** instead of raw user-turn concatenation —
  C7 (multi-faceted bilingual healthcare) would benefit.
- **Diversity boost (MMR)** in retrieval to avoid product-family clustering
  (C9 originally returned 5 Java variants and crowded out AWS/Docker/SQL).
- **Holdout evaluation** with a simulated-user LLM (the spec describes how
  SHL runs this); I could only verify against the 10 public traces verbatim.
