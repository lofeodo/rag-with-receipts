"""Combines the NLI entailment signal and the lexical overlap signal into a single
grounded/ungrounded/contradicted verdict per citation (Step 6).

Label logic:
  contradiction_prob >= contradiction_threshold          -> "contradicted"
  elif entailment_prob >= entailment_threshold
       or overlap_score >= overlap_threshold              -> "grounded"
  else                                                     -> "ungrounded"

The OR-combination is deliberate: overlap catches near-verbatim support a
general-domain NLI model (trained on SNLI/MNLI-style data, not OSRS wiki jargon) might
misjudge as neutral; NLI catches correctly-paraphrased support with low lexical overlap.
Contradiction is checked first and wins regardless of overlap, since a citation whose
chunk actively contradicts its own claim is a worse failure than a merely-unsupported one
and shouldn't be masked by high lexical overlap with the wrong claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_receipts.generation.models import Citation
from rag_receipts.grounding.config import GroundingConfig
from rag_receipts.grounding.entailment import EntailmentChecker
from rag_receipts.grounding.models import AnswerGrounding, GroundingLabel, GroundingVerdict
from rag_receipts.grounding.overlap import token_overlap
from rag_receipts.ingestion.utils import breadcrumb


def _label(
    entailment_prob: float,
    contradiction_prob: float,
    overlap_score: float,
    config: GroundingConfig,
) -> GroundingLabel:
    if contradiction_prob >= config.contradiction_threshold:
        return "contradicted"
    if entailment_prob >= config.entailment_threshold or overlap_score >= config.overlap_threshold:
        return "grounded"
    return "ungrounded"


@dataclass
class GroundingChecker:
    config: GroundingConfig
    entailment: EntailmentChecker

    @classmethod
    def from_config(cls, config: GroundingConfig) -> "GroundingChecker":
        return cls(config=config, entailment=EntailmentChecker.from_config(config))

    def check(self, citations: list[Citation]) -> list[GroundingVerdict]:
        if not citations:
            return []

        pairs = [
            (breadcrumb(c.chunk.page_title, c.chunk.section_path) + "\n\n" + c.chunk.text, c.claim)
            for c in citations
        ]
        scores = self.entailment.score(pairs)  # (n, 3): entailment, contradiction, neutral

        verdicts: list[GroundingVerdict] = []
        for citation, row in zip(citations, scores):
            entailment_prob, contradiction_prob, neutral_prob = (float(x) for x in row)
            overlap_score = token_overlap(citation.claim, citation.chunk.text)
            label = _label(entailment_prob, contradiction_prob, overlap_score, self.config)
            verdicts.append(
                GroundingVerdict(
                    chunk_id=citation.chunk_id,
                    claim=citation.claim,
                    entailment_prob=entailment_prob,
                    contradiction_prob=contradiction_prob,
                    neutral_prob=neutral_prob,
                    overlap_score=overlap_score,
                    label=label,
                )
            )
        return verdicts


def summarize_grounding(verdicts: list[GroundingVerdict]) -> AnswerGrounding:
    return AnswerGrounding(
        citation_verdicts=verdicts,
        all_grounded=all(v.label == "grounded" for v in verdicts),
        any_contradicted=any(v.label == "contradicted" for v in verdicts),
    )
