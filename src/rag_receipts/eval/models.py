"""Result shapes for the eval harness (Step 5, extended in Step 6): gold questions,
retrieval scores, judge verdicts, grounding verdicts, and the per-question / aggregate
report shapes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from rag_receipts.generation.models import GeneratedAnswer
from rag_receipts.grounding.models import AnswerGrounding

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
    type: QuestionType | None  # None only when answerable is False - "what type of
    # question would this have been" isn't a meaningful label for one that's out of
    # corpus by design, so it's excluded from the type-bucketed breakdowns rather than
    # forced into single_hop/multi_hop
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
    grounding: AnswerGrounding | None = None  # None when generation raised or the answer
    # had no real citations to check (e.g. an answerable=False response)
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
    grounding: dict = field(default_factory=dict)
    grounding_by_type: dict = field(default_factory=dict)
    grounding_by_correctness_label: dict = field(default_factory=dict)
