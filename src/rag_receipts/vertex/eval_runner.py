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
from rag_receipts.vertex.answer_client import VertexAnswerer
from rag_receipts.vertex.models import VertexEvalResult
from rag_receipts.vertex.search_client import VertexRetriever


def _dedupe_cited_chunks(generated: GeneratedAnswer) -> list:
    """score_retrieval()/recall_at_k() assume a deduplicated retrieved-chunk
    list (true for Retriever/VertexRetriever, which each return at most one
    RetrievedChunk per distinct chunk_id) - but Vertex's Answer API legitimately
    cites the same chunk_id multiple times, once per claim/sentence it
    supports. Confirmed live (2026-10-02): q002 cited one chunk 8 times across
    different sentences, producing recall > 1.0 (8/1) before this fix. Dedupes
    by chunk_id, keeping first-occurrence order so MRR still reflects the
    earliest cited position."""
    seen: set[str] = set()
    deduped = []
    for citation in generated.citations:
        if citation.chunk_id in seen:
            continue
        seen.add(citation.chunk_id)
        deduped.append(citation.chunk)
    return deduped


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


def run_vertex_eval_answer(
    questions: list[EvalQuestion],
    *,
    vertex_answerer: VertexAnswerer,
    judge: Judge,
    grounding_checker: GroundingChecker,
) -> list[VertexEvalResult]:
    """Config B: no Retriever/Generator call at all - Vertex's Answer API owns
    retrieval+generation end-to-end. Retrieval precision is scored against the
    *distinct* chunk_ids Vertex actually cited (via _dedupe_cited_chunks - a
    cited chunk can and does repeat across claims, see that function's
    docstring), not a separate ranked search-results list - the Answer API
    doesn't expose one the way the Search API does, so "what was retrieved"
    here means "what was cited," a narrower signal than config A/baseline's
    full top-k. This asymmetry must be stated wherever config B's retrieval
    numbers are reported (see compare_vertex.py's methodology_notes)."""
    results: list[VertexEvalResult] = []
    for question in questions:
        try:
            generated, vertex_grounding_score = vertex_answerer.answer(question.question)
        except RuntimeError as exc:
            results.append(
                VertexEvalResult(
                    question=question,
                    retrieval=None,
                    generated=None,
                    judge=None,
                    correctness_label="error",
                    grounding=None,
                    error=f"vertex answer failed: {exc}",
                    vertex_grounding_score=None,
                )
            )
            continue

        retrieval_score = (
            score_retrieval(_dedupe_cited_chunks(generated), question.gold_chunk_ids)
            if question.answerable
            else None
        )
        grounding = _compute_grounding(generated, grounding_checker)

        if question.answerable and not generated.answerable:
            results.append(
                VertexEvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="incorrectly_abstained",
                    grounding=grounding,
                    vertex_grounding_score=vertex_grounding_score,
                )
            )
            continue

        if not question.answerable and generated.answerable:
            results.append(
                VertexEvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="incorrectly_answered",
                    grounding=grounding,
                    vertex_grounding_score=vertex_grounding_score,
                )
            )
            continue

        if not question.answerable and not generated.answerable:
            results.append(
                VertexEvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="correct_abstention",
                    grounding=grounding,
                    vertex_grounding_score=vertex_grounding_score,
                )
            )
            continue

        # Both sides say "yes" - the only case that actually spends a judge call.
        assert question.gold_answer is not None  # enforced by load_eval_questions
        try:
            verdict = judge.score(question.question, question.gold_answer, generated.answer)
        except RuntimeError as exc:
            results.append(
                VertexEvalResult(
                    question=question,
                    retrieval=retrieval_score,
                    generated=generated,
                    judge=None,
                    correctness_label="error",
                    grounding=grounding,
                    error=f"judging failed: {exc}",
                    vertex_grounding_score=vertex_grounding_score,
                )
            )
            continue

        results.append(
            VertexEvalResult(
                question=question,
                retrieval=retrieval_score,
                generated=generated,
                judge=verdict,
                correctness_label=verdict.verdict,
                grounding=grounding,
                vertex_grounding_score=vertex_grounding_score,
            )
        )

    return results
