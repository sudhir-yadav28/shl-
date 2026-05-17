# SHL Conversational Assessment Recommender — Approach

**Public endpoint:** https://shl-recommender-ab5b.onrender.com
**Repo:** https://github.com/sudhir-yadav28/shl-
**Mean Recall@10 on the 10 public traces: 0.94**

## How I read the problem

Before writing any code I went through all 10 sample conversations. Two things stood out. First, "good" recommendations almost always include some canonical SHL products — OPQ32r, Verify G+, Global Skills Assessment — even when the user never mentions them by name. Second, the agent's job is to know *when* to do what: ask, retrieve, refine, compare, or refuse. The rubric weights schema compliance and catalog-only URLs as hard fails, and the rest is Recall@10 plus behaviour probes. That shaped every design choice that followed.

I considered three architectures: a pure LLM-with-tools agent (e.g. give it a `search_catalog` tool and let it loop), a LangGraph state machine, and a thin orchestrator with one LLM call per turn. I went with the third. Tool-use is good when reasoning spans many steps and the action space is large — here it's five actions and the cost of a wrong one shows directly in the score. A small orchestrator is easier to defend in an interview and easier to test.

## What `/chat` does per turn

1. **Regex pre-check** catches obvious prompt-injection (`ignore previous instructions`, `act as`, etc.) and short-circuits with a refusal. Faster than burning an LLM call on something I already know is bad input.
2. **Embed the query** (Gemini `text-embedding-001`, 3072-dim). The query is the concatenation of all user turns so refines and compares get full context for free.
3. **Hybrid retrieval** — BM25 over name + description + keys + job levels, plus dense cosine. Fused via Reciprocal Rank Fusion (k=60). Top 18.
4. **Canonical injection** — append a small set of always-relevant SHL products (OPQ32r, Verify G+, GSA, OPQ UCR 2.0, Smart Interview Live Coding, etc.) so the LLM can choose them even when retrieval ranks them low.
5. **One LLM call** classifies the action, drafts the reply, and picks names from the candidate block. Output is JSON-mode so parsing is deterministic.
6. **Shortlist builder** assembles the final 10 items in priority order: tech-keyword force-includes → trigger-matched canonicals → LLM picks → retrieval rank.
7. **Pydantic validator** checks every URL against the catalog and drops anything I can't ground. The schema is the last thing the response touches.

## What lifted the score and how I measured it

I built a replay harness early — `eval/replay.py` parses each `Cn.md`, replays the user turns against the live endpoint, and computes Recall@10 against the labelled URLs. The first working agent scored 0.11. The path to 0.94 was three changes.

**Fill the shortlist to 10.** On multi-turn refines the LLM was returning only 1–3 names. Recall@10 has no precision penalty, so under-filling is pure score loss. I added a top-up that fills empty slots from retrieval rank. Recall went 0.11 → 0.43. Big win.

**Conditional canonical force-include.** Looking at C5 (sales restructuring) the expected shortlist included OPQ32r and GSA — neither of which appears in BM25/dense top-10 for a "sales audit" query because their descriptions don't say "sales". I added a small list of canonical products that get force-included when query triggers match (OPQ32r is unconditional, Verify G+ requires "senior" or "graduate", GSA requires "audit" or "re-skill", etc.). 0.43 → 0.61. The conditional part matters — forcing every canonical on every query hurt C6 (plant-operator safety) by displacing the right product.

**Tech-keyword force-include.** C9 had the user explicitly naming AWS, Docker, SQL, but the retriever was returning five Java variants and crowding them out. I added a small hand-curated map from user keywords to exact catalog products (`docker` → Docker (New), `aws` → Amazon Web Services (AWS) Development (New), etc.). 0.61 → 0.94.

Per-trace Recall@10 on the public set:

| C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 0.80 | 0.75 | 1.00 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **0.94** |

Behaviour probes live in `tests/test_behavior_probes.py` — nine scripted conversations with binary assertions covering schema shape, no-recommend-on-vague-turn-1, prompt-injection refusal, off-topic refusal, refine keeps prior items, end-of-conversation only on user confirmation, and test_type as a letter code. All nine pass.

## Stack and why

**FastAPI + Pydantic v2** because the schema is non-negotiable. Pydantic is the last gate before any response leaves the service — off-schema returns a 5xx, never silently malformed JSON.

**Gemini 2.0 Flash as primary, Groq `llama-3.1-8b-instant` as fallback.** I started on Gemini 2.5 Flash and burned through its 250-RPD free quota in a single eval run. Switched to 2.0 Flash (1500 RPD) and added Groq so failures on one provider transparently route to the other. Two separate quotas matters more in practice than I expected.

**Gemini `text-embedding-001`** embedded the catalog once offline. The result is a 4.5 MB `embeddings.npy` shipped in the repo. Cold start on Render is 3 seconds (load + index build) instead of about 7 minutes (re-embed 377 docs over the network).

**`rank_bm25` for lexical, NumPy matmul over 377 vectors for dense.** I had originally planned FAISS and `sentence-transformers/all-MiniLM-L6-v2`, but PyTorch is ~500 MB and would have blown Render's 512 MB RAM cap. Switching to Gemini embeddings made the deployment ~270 MB total.

**Render free Docker tier.** The spec's 2-minute cold-start allowance for `/health` is essentially written for it.

## What I tried that didn't work

I spent a couple of hours on prompt engineering before realising the LLM wasn't the problem on Recall — it was that the LLM was picking *fewer* items than the budget allowed, and the retriever was missing canonical products entirely. No amount of prompt tuning fixes that. The shortlist builder did.

I also tried using only the latest user message as the retrieval query. That broke refines ("add personality" alone has no signal). Concatenating all user turns is the cheap fix that gives the same context the LLM sees.

The first version of the canonical injection forced every canonical on every query. C6 dropped from 1.00 to 0.50 because canonicals displaced the right safety product. Making the inclusion conditional on query triggers fixed it.

## Resilience

If both Gemini and Groq fail (both rate-limited or network outage), the agent falls through to a retrieval-only path. The shortlist still populates via tech-keyword + canonical + retrieval rank, so Recall@10 is preserved even without an LLM. A small signal-term heuristic in the fallback path decides between clarify and recommend so it doesn't violate the no-recommend-on-vague-turn-1 probe.

## What I'd add next

LLM-based query rewriting before retrieval, mostly to help C7 (multi-faceted bilingual-healthcare-admin) which still relies too much on the hand-curated keyword layer. MMR-style diversity inside the retriever to drop that hand layer entirely. And a simulated-user harness that uses an LLM as the user, the way SHL actually runs evaluation, so I could test holdout-style queries instead of replaying the public traces verbatim.

## AI tool usage

I used Claude (via Claude Code) as a pair-programmer throughout — code generation, debugging the rate-limit cascade when both providers got exhausted, and helping me diagnose why C8 and C9 were under-performing. The design calls (orchestrator over tool-use, canonical injection, the tech-keyword layer, the resilience path) and all the prompts and keyword lists are mine. I verified every change against the 10 public traces and the probe suite before committing.
