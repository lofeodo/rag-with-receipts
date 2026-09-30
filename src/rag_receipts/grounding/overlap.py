"""Lexical overlap: a cheap, model-free grounding signal (Step 6).

Complements the NLI entailment check in entailment.py - catches near-verbatim support
that a general-domain NLI model (trained on SNLI/MNLI-style data, not OSRS wiki jargon)
might misjudge as neutral.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def token_overlap(claim: str, chunk_text: str) -> float:
    """|claim_tokens ∩ chunk_tokens| / |claim_tokens|. Returns 0.0 for an empty claim
    (no tokens to be grounded, so no basis for a positive overlap claim)."""
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    return len(claim_tokens & chunk_tokens) / len(claim_tokens)
