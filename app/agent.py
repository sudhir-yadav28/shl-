"""The agent orchestrator. Turns a conversation history into a ChatResponse.

Design:
- 1 retrieval call (local, fast)
- 1 query-embedding call (Gemini)
- 1 LLM decision call (Gemini, JSON output)
- Hard validators on the way out: only catalog URLs survive.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.catalog import Catalog, Product
from app.config import embed_query, gemini_model
from app.guardrails import REFUSAL_INJECTION, looks_like_injection
from app.prompts import (
    DECISION_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    format_candidates_block,
    format_conversation,
)
from app.retrieval import HybridIndex
from app.schemas import ChatResponse, Message, Recommendation

logger = logging.getLogger("agent")

RETRIEVAL_TOP_K = 30
MAX_RECOMMENDATIONS = 10
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


@dataclass
class AgentDeps:
    catalog: Catalog
    index: HybridIndex


def build_query(messages: list[Message]) -> str:
    """Concatenate all user turns. Refine/compare get full context for free."""
    return " ".join(m.content for m in messages if m.role == "user").strip()


def _extract_json(text: str) -> dict:
    """LLMs occasionally wrap JSON in markdown fences. Strip and parse the first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = JSON_BLOCK_RE.search(text)
        if m:
            return json.loads(m.group(0))
        raise


def _safe_response(reply: str, end_of_conversation: bool = False) -> ChatResponse:
    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=end_of_conversation,
    )


def respond(messages: list[Message], deps: AgentDeps) -> ChatResponse:
    if not messages:
        return _safe_response("I need a message to respond to.")
    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if last_user is None:
        return _safe_response("I'm here to help with SHL assessments. What role are you hiring for?")

    if looks_like_injection(last_user.content):
        logger.info("guardrail: injection pattern matched")
        return _safe_response(REFUSAL_INJECTION)

    query = build_query(messages)
    try:
        qvec = embed_query(query)
    except Exception as e:
        logger.warning(f"embedding failed, falling back to BM25-only: {e}")
        qvec = None
    candidates = deps.index.hybrid(query, qvec, top_k=RETRIEVAL_TOP_K)

    if not candidates:
        return _safe_response(
            "I couldn't find catalog matches for that. Could you tell me the role, skills, "
            "or assessment area you're focused on?"
        )

    prompt = DECISION_PROMPT_TEMPLATE.format(
        conversation=format_conversation(messages),
        candidates=format_candidates_block(candidates),
    )

    try:
        model = gemini_model()
        result = model.generate_content(
            [SYSTEM_PROMPT, prompt],
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "max_output_tokens": 1024,
            },
        )
        decision = _extract_json(result.text)
    except Exception as e:
        logger.exception(f"LLM call failed: {e}")
        return _safe_response(
            "Something went wrong on my end. Could you tell me the role or skills "
            "you're assessing for?"
        )

    action = decision.get("action", "clarify")
    reply = (decision.get("reply") or "").strip() or "Could you tell me more about the role?"
    names = decision.get("recommendation_names") or []
    end = bool(decision.get("end_of_conversation", False))

    recs: list[Recommendation] = []
    if action in {"recommend", "refine", "compare"}:
        candidate_names = {p.name.lower() for p, _ in candidates}
        for n in names:
            if not isinstance(n, str):
                continue
            n_clean = n.strip()
            if n_clean.lower() not in candidate_names:
                # Try direct catalog lookup as a forgiving fallback
                p = deps.catalog.find_by_name(n_clean)
                if p is None:
                    logger.info(f"dropped hallucinated/unknown name: {n_clean!r}")
                    continue
            else:
                p = deps.catalog.find_by_name(n_clean)
            if p is None:
                continue
            if any(r.url == p.url for r in recs):
                continue
            recs.append(Recommendation(name=p.name, url=p.url, test_type=p.test_type))
            if len(recs) >= MAX_RECOMMENDATIONS:
                break

    if action in {"recommend", "refine"} and not recs:
        action = "clarify"
        reply = (
            "I'm not seeing enough signal yet. Could you tell me the role, "
            "seniority, or specific skills you're assessing for?"
        )

    if action in {"clarify", "refuse"}:
        recs = []
        end = False

    return ChatResponse(reply=reply, recommendations=recs, end_of_conversation=end)
