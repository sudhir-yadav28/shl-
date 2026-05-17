"""Cheap pre-LLM checks. Catches obvious prompt-injection and trivially off-topic
input before we spend an LLM call. Everything subtler goes to the LLM, which
has refusal logic in its system prompt.
"""

from __future__ import annotations

import re

# Patterns that indicate prompt-injection or jailbreak attempts.
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:a|an)\b", re.I),
    re.compile(r"\bact\s+as\s+(?:a|an)\b.*\b(?:hacker|admin|system|root)\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\breveal\s+(?:your|the)\s+(?:prompt|instructions|system)", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bdan\s+mode\b", re.I),
]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


REFUSAL_INJECTION = (
    "I can only help with selecting SHL assessments from the catalog. "
    "What role or skills are you assessing for?"
)

REFUSAL_OFF_TOPIC = (
    "I can only help with SHL assessment selection. "
    "Tell me the role, skills, or seniority you're hiring for and I'll recommend assessments."
)
