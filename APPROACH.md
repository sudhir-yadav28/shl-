# SHL Conversational Assessment Recommender — Approach

**Public endpoint:** https://shl-recommender-ab5b.onrender.com
**Repo:** https://github.com/sudhir-yadav28/shl-

---

## How I thought about the problem

The brief reads like a search problem, but it isn't really one. A keyword search over the SHL catalog already exists on the SHL site — the value of this assignment is in the *conversation*, not the lookup. So I framed the system around three constraints the rubric makes explicit: every response has to match the schema exactly, every URL has to come from the catalog (no hallucinations), and the final shortlist has to recover the items a domain expert would pick. Schema and grounding are things I can guarantee in code; recommendation quality is the messy part where the LLM helps.

That framing pushed me away from a "give the LLM tools and let it figure things out" design. Tool-use is great when the agent has to reason across many steps, but here the steps are simple — clarify or recommend or refine or compare or refuse — and the cost of getting any of them wrong shows up directly in the score. I went with a lightweight orchestrator: deterministic retrieval, one LLM call per turn for the natural-language work, and a post-processor that enforces the schema and URL whitelist. Each component does one thing I can defend.

## What happens on each `/chat` call

A request walks through five stages. First, a small regex catches obvious prompt-injection attempts and short-circuits with a refusal — I'd rather refuse fast than pay for an LLM call on `ignore previous instructions`. Second, the latest user turn is embedded via Gemini's `text-embedding-001` and joined with all prior user messages to form the retrieval query. Third, a hybrid retriever (BM25 over name + description + keys, plus dense cosine over the embedding index) returns its top 18 candidates, fused by Reciprocal Rank Fusion. To this list I append a small set of "canonical" SHL products — OPQ32r, Verify G+, GSA, and a few others — that show up across many hiring contexts but whose descriptions don't always lexically match the user's specific phrasing. Fourth, one LLM call classifies the action, drafts the reply, and picks names from the candidate block; output is JSON-mode so parsing is deterministic. Fifth, a shortlist builder assembles the final ten items in a deliberate priority order, and a Pydantic validator drops anything that isn't a real catalog URL.

The shortlist builder is where most of the score came from, so it's worth explaining. The order is: tech-keyword force-includes first (e.g. if the user typed "docker", `Docker (New)` gets a guaranteed slot), then trigger-matched canonicals (OPQ32r unconditional, Verify G+ when "senior" or "graduate" appears, GSA when "audit" or "re-skill" appears, and so on), then the LLM's own picks, and finally the retrieval-ranked top-up to fill ten slots. Recall@10 has no precision penalty, so under-filling is pure score loss — and in my early traces the LLM frequently returned only one to three names on refine turns. The priority ordering also defends well: each slot's presence has a reason ("the user named this technology", "this is the standard SHL personality measure for senior selection", "this is what semantic + lexical retrieval ranked highest"), not "the model picked it".

## Where the design got tested

I built a replay harness that parses each of the ten provided conversation traces for the user turns and the labeled final shortlist, replays the conversation against the live endpoint, and computes Recall@10 per trace. Mean recall starts at 0.11 with a naive setup and lands at 0.94 after the changes above. The biggest jumps came from filling the shortlist to ten (recall lift +0.40), forcing canonicals like OPQ32r conditional on context (+0.18), and the tech-keyword force-include layer (+0.16). C9 was particularly instructive: the user names seven technologies in a JD, but the retriever was returning five Java variants and crowding out AWS, Docker, and SQL. The tech-keyword path catches exactly that case.

Behavior probes live in `tests/test_behavior_probes.py` — nine scripted conversations with binary assertions covering schema shape, no-recommend-on-vague-turn-1, prompt-injection refusal, off-topic refusal, refine keeps prior items, end-of-conversation only on explicit confirmation, and test-type letter codes. They all pass and I run them on every change.

## Stack and why

FastAPI plus Pydantic v2 because the schema is non-negotiable and Pydantic gives me the last gate before any response leaves the service. Gemini 2.0 Flash as the primary LLM (15 RPM / 1500 RPD on the free tier — comfortable headroom for evaluation) with Groq's `llama-3.1-8b-instant` as a transparent fallback when Gemini errors. The two providers sit on separate quotas, which mattered during development when I burned my Gemini daily limit pretty quickly. Embeddings are Gemini's `text-embedding-001`, computed offline once and shipped as a 4.5 MB `embeddings.npy` so cold start on Render is three seconds instead of seven minutes. Dense search is a NumPy matmul over 377 vectors — FAISS would be overkill at this scale. Lexical search is `rank_bm25`. Deployment is Render's free Docker tier; the assignment's two-minute cold-start allowance is essentially written for it.

## What didn't work, and what I'd add with more time

The first attempt at the agent was a single LLM call with no post-processing top-up. It plateaued around 0.43 Mean Recall@10. The model consistently under-filled the shortlist (one to three items even on multi-turn refines) and sometimes picked similar-sounding products instead of the canonical core ones — Enterprise Leadership Report 2.0 in place of OPQ Leadership Report, for instance. The priority-ordered shortlist builder is what recovered most of the gap; the LLM is doing the language and the judgment about *which* items, but it's no longer responsible for *how many* or for ensuring the canonical defaults make the list. I also tried `sentence-transformers` locally first, but the PyTorch dependency was about 500 MB and would have blown Render's 512 MB RAM cap — Gemini embeddings keep the deployment lean.

A few things I'd add given more time. The retrieval would benefit from MMR-style diversity so I could drop the tech-keyword layer (which works but is hand-curated). An LLM-based query rewriter before retrieval would help on multi-faceted queries like C7 (bilingual + healthcare + admin + Spanish), where the concatenated user turns confuse BM25. And I'd build a simulated-user harness — the spec describes how SHL runs evaluation against my endpoint with an LLM playing the user — so I could test holdout-style queries instead of just replaying the public traces verbatim.

## AI tool usage disclosure

I used Claude as a coding assistant throughout — pair-debugging retrieval gaps, diagnosing the rate-limit cascade when both providers got exhausted, and proposing the shortlist-builder priority ordering after I described what was happening on the C8 and C9 traces. The design calls (single-LLM-call orchestrator over tool-use, canonical injection, the tech-keyword layer, the resilience path when both LLMs are down) and the prompt content and keyword lists are mine. I verified every change against the ten public traces and the behavior probe suite before committing.
