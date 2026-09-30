"""Orchestrates retrieve + generate + judge + grounding-check over a gold Q/A set and
aggregates results (Step 5, extended in Step 6).

Per-question flow:
  1. retrieve - always run, scored against gold_chunk_ids when the question is
     answerable=True (skipped, RetrievalScore=None, for answerable=False questions -
     there's no gold evidence set to score against).
  2. generate - Generator.generate() can raise RuntimeError (max_tokens truncation, a
     missing tool_use block); caught here and recorded as an "error" result rather than
     aborting the whole batch run.
  3. grounding check - run once per generated answer (if it has any real citations) via
     GroundingChecker, independent of the correctness-label branch below: every citation
     in a real GeneratedAnswer was already resolved against real chunk_ids in Step 4, so
     there's no reason to skip checking it just because the answer turned out to be an
     incorrectly-abstained/incorrectly-answered case, or because judging itself later
     failed.
  4. correctness resolution - three of the six non-error labels are resolved directly
     from the answerable flags with no judge call spent (correct_abstention /
     incorrectly_abstained / incorrectly_answered); only when both sides say "yes" does
     the judge actually grade the answer against gold_answer (correct / partially_correct
     / incorrect). Judge.score() can also raise RuntimeError, caught the same way.
"""

from __future__ import annotations

from rag_receipts.eval.judge import Judge
from rag_receipts.eval.models import EvalQuestion, EvalResult, EvalSummary
from rag_receipts.eval.retrieval_metrics import score_retrieval
from rag_receipts.generation.generator import Generator
from rag_receipts.generation.models import GeneratedAnswer
from rag_receipts.grounding.checker import GroundingChecker, summarize_grounding
from rag_receipts.grounding.models import AnswerGrounding
from rag_receipts.retrieval.pipeline import Retriever


def _compute_grounding(
    generated: GeneratedAnswer | None, grounding_checker: GroundingChecker
) -> AnswerGrounding | None:
    if generated is None or not generated.citations:
        return None
    return summarize_grounding(grounding_checker.check(generated.citations))


def run_eval(
    questions: list[EvalQuestion],
    *,
    retriever: Retriever,
    generator: Generator,
    judge: Judge,
    grounding_checker: GroundingChecker,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    for question in questions:
        chunks = retriever.retrieve(question.question)
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


_CORRECT_LABELS = {"correct", "correct_abstention"}


def _retrieval_stats(results: list[EvalResult]) -> dict:
    scores = [r.retrieval for r in results if r.retrieval is not None]
    if not scores:
        return {"mean_precision": 0.0, "mean_recall": 0.0, "mean_mrr": 0.0, "hit_rate": 0.0, "count": 0}
    n = len(scores)
    return {
        "mean_precision": sum(s.precision for s in scores) / n,
        "mean_recall": sum(s.recall for s in scores) / n,
        "mean_mrr": sum(s.mrr for s in scores) / n,
        "hit_rate": sum(1 for s in scores if s.hit) / n,
        "count": n,
    }


def _correctness_stats(results: list[EvalResult]) -> dict:
    non_error = [r for r in results if r.correctness_label != "error"]
    if not non_error:
        return {"accuracy": 0.0, "count": 0}
    correct = sum(1 for r in non_error if r.correctness_label in _CORRECT_LABELS)
    return {"accuracy": correct / len(non_error), "count": len(non_error)}


def _grounding_stats(results: list[EvalResult]) -> dict:
    groundings = [r.grounding for r in results if r.grounding is not None]
    if not groundings:
        return {
            "grounded_rate": 0.0,
            "contradicted_rate": 0.0,
            "ungrounded_rate": 0.0,
            "fully_grounded_answer_rate": 0.0,
            "citation_count": 0,
            "answer_count": 0,
        }
    verdicts = [v for g in groundings for v in g.citation_verdicts]
    n_citations = len(verdicts)
    n_answers = len(groundings)
    return {
        "grounded_rate": sum(1 for v in verdicts if v.label == "grounded") / n_citations,
        "contradicted_rate": sum(1 for v in verdicts if v.label == "contradicted") / n_citations,
        "ungrounded_rate": sum(1 for v in verdicts if v.label == "ungrounded") / n_citations,
        "fully_grounded_answer_rate": sum(1 for g in groundings if g.all_grounded) / n_answers,
        "citation_count": n_citations,
        "answer_count": n_answers,
    }


def summarize(results: list[EvalResult]) -> EvalSummary:
    # None (unanswerable questions with no meaningful type) is excluded from the
    # type-bucketed breakdowns - those questions still count in the overall stats above.
    types = sorted({r.question.type for r in results if r.question.type is not None})
    labels = sorted({r.correctness_label for r in results})

    label_counts: dict[str, int] = {}
    for r in results:
        label_counts[r.correctness_label] = label_counts.get(r.correctness_label, 0) + 1

    return EvalSummary(
        question_count=len(results),
        retrieval=_retrieval_stats(results),
        retrieval_by_type={t: _retrieval_stats([r for r in results if r.question.type == t]) for t in types},
        correctness_accuracy=_correctness_stats(results)["accuracy"],
        correctness_by_type={
            t: _correctness_stats([r for r in results if r.question.type == t]) for t in types
        },
        correctness_label_counts=label_counts,
        error_count=sum(1 for r in results if r.correctness_label == "error"),
        grounding=_grounding_stats(results),
        grounding_by_type={t: _grounding_stats([r for r in results if r.question.type == t]) for t in types},
        grounding_by_correctness_label={
            label: _grounding_stats([r for r in results if r.correctness_label == label]) for label in labels
        },
    )
