"""Result shape for config B (Vertex's own Answer API) - Step detail: Vertex AI
Search comparison stretch goal.

VertexEvalResult is EvalResult's fields plus vertex_grounding_score, Vertex's
own self-reported groundedness signal, captured alongside our own
GroundingChecker's verdict on the same citations for a side-by-side report.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_receipts.eval.models import CorrectnessLabel, EvalQuestion, JudgeVerdict, RetrievalScore
from rag_receipts.generation.models import GeneratedAnswer
from rag_receipts.grounding.models import AnswerGrounding


@dataclass
class VertexEvalResult:
    question: EvalQuestion
    retrieval: RetrievalScore | None
    generated: GeneratedAnswer | None
    judge: JudgeVerdict | None
    correctness_label: CorrectnessLabel
    grounding: AnswerGrounding | None = None
    error: str | None = None
    vertex_grounding_score: float | None = None
