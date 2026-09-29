"""Result shapes for the eval harness (Step 5): gold questions, retrieval scores,
judge verdicts, and the per-question / aggregate report shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_receipts.generation.models import GeneratedAnswer

QuestionType = Literal["single_hop", "multi_hop"]
Verdict = Literal["correct", "partially_correct", "incorrect"]

# A question's correctness is resolved one of two ways:
# - the expected vs actual `answerable` flags mismatch or both correctly say "no" -
#   resolved directly, no judge call spent (correct_abstention / incorrectly_abstained /
#   incorrectly_answered)
# - both say "yes" - the judge scores the generated answer against gold_answer
#   (correct / partially_correct / incorrect)
# "error" covers a question where generation or judging raised.
CorrectnessLabel = Literal[
    "correct",
    "partially_correct",
    "incorrect",
    "correct_abstention",
    "incorrectly_abstained",
    "incorrectly_answered",
    "error",
]


@dataclass
class EvalQuestion:
    """One hand-authored gold Q/A pair from data/eval/qa_pairs.json."""

    id: str
    question: str
    type: QuestionType
    answerable: bool  # expected GeneratedAnswer.answerable
    gold_chunk_ids: list[str]  # empty when answerable is False
    gold_answer: str | None  # reference answer; None when answerable is False


@dataclass
class RetrievalScore:
    precision: float
    recall: float
    mrr: float
    hit: bool
    retrieved_chunk_ids: list[str]


@dataclass
class JudgeVerdict:
    verdict: Verdict
    reasoning: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


@dataclass
class EvalResult:
    question: EvalQuestion
    retrieval: RetrievalScore | None  # None when the question is answerable=False
    generated: GeneratedAnswer | None  # None if generation raised
    judge: JudgeVerdict | None  # None when resolved without a judge call, or generation raised
    correctness_label: CorrectnessLabel
    error: str | None = None  # populated when generation or judging raised


@dataclass
class EvalSummary:
    question_count: int
    retrieval: dict
    retrieval_by_type: dict
    correctness_accuracy: float
    correctness_by_type: dict
    correctness_label_counts: dict[str, int]
    error_count: int
