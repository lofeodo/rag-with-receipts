"""Parallel eval runners for the Vertex AI Search comparison stretch goal.

run_vertex_eval_search (config A) is a parallel copy of eval/runner.py::run_eval()'s
body, swapping the hand-built Retriever for VertexRetriever - not a forced
Protocol over both, per the project's own established convention of duplicated
runners/fakes over cross-package abstraction (see tests/eval/helpers.py vs
tests/generation/helpers.py). It reuses score_retrieval, Generator.generate,
Judge.score, and GroundingChecker.check/summarize_grounding directly and
returns the *existing* EvalResult/EvalSummary types, so eval/runner.py's own
summarize() works on config A's results unmodified.
"""

from __future__ import annotations

from rag_receipts.eval.judge import Judge
from rag_receipts.eval.models import EvalQuestion, EvalResult
from rag_receipts.eval.retrieval_metrics import score_retrieval
from rag_receipts.generation.generator import Generator
from rag_receipts.generation.models import GeneratedAnswer
from rag_receipts.grounding.checker import GroundingChecker, summarize_grounding
from rag_receipts.grounding.models import AnswerGrounding
from rag_receipts.vertex.search_client import VertexRetriever


def _compute_grounding(
    generated: GeneratedAnswer | None, grounding_checker: GroundingChecker
) -> AnswerGrounding | None:
    if generated is None or not generated.citations:
        return None
    return summarize_grounding(grounding_checker.check(generated.citations))


def run_vertex_eval_search(
    questions: list[EvalQuestion],
    *,
    vertex_retriever: VertexRetriever,
    generator: Generator,
    judge: Judge,
    grounding_checker: GroundingChecker,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for question in questions:
        chunks = vertex_retriever.retrieve(question.question)
        retrieval_score = (
            score_retrieval(chunks, question.gold_chunk_ids) if question.answerable else None
        )

        try:
            generated = generator.generate(question.question, chunks)
        except RuntimeError as exc:
            results.append(
                EvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=None,
                    judge=None,
                    correctness_label="error",
                    grounding=None,
                    error=f"generation failed: {exc}",
                )
            )
            continue

        grounding = _compute_grounding(generated, grounding_checker)

        if question.answerable and not generated.answerable:
            results.append(
                EvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="incorrectly_abstained",
                    grounding=grounding,
                )
            )
            continue

        if not question.answerable and generated.answerable:
            results.append(
                EvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="incorrectly_answered",
                    grounding=grounding,
                )
            )
            continue

        if not question.answerable and not generated.answerable:
            results.append(
                EvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="correct_abstention",
                    grounding=grounding,
                )
            )
            continue

        # Both sides say "yes" - the only case that actually spends a judge call.
        assert question.gold_answer is not None  # enforced by load_eval_questions
        try:
            verdict = judge.score(question.question, question.gold_answer, generated.answer)
        except RuntimeError as exc:
            results.append(
                EvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="error",
                    grounding=grounding,
                    error=f"judging failed: {exc}",
                )
            )
            continue

        results.append(
            EvalResult(
                question=question,
                retrieval=retrieval_score,
                generated=generated,
                judge=verdict,
                correctness_label=verdict.verdict,
                grounding=grounding,
            )
        )

    return results
