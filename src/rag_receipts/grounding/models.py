"""Result shapes for the grounding/hallucination check (Step 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GroundingLabel = Literal["grounded", "ungrounded", "contradicted"]


@dataclass
class GroundingVerdict:
    """Per-citation grounding verdict: does the cited chunk actually support the claim
    attached to it? Distinct from Step 4's hallucinated_citation_ids, which only checks
    whether the chunk_id was in the retrieved set at all - every citation checked here
    already resolved to a real chunk."""

    chunk_id: str
    claim: str
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    overlap_score: float
    label: GroundingLabel


@dataclass
class AnswerGrounding:
    citation_verdicts: list[GroundingVerdict]
    all_grounded: bool  # True iff every verdict is "grounded" (vacuously True if empty)
    any_contradicted: bool
