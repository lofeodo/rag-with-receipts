from __future__ import annotations

from rag_receipts.eval.models import EvalQuestion, JudgeVerdict
from rag_receipts.generation.models import Citation, GeneratedAnswer
from rag_receipts.grounding.models import GroundingVerdict
from rag_receipts.vertex.eval_runner import run_vertex_eval_search

from .helpers import FakeGenerator, FakeGroundingChecker, FakeJudge, FakeVertexRetriever, make_chunks


def _grounding_verdict(chunk_id: str, label: str) -> GroundingVerdict:
    return GroundingVerdict(
        chunk_id=chunk_id,
        claim="some claim",
        entailment_prob=0.9 if label == "grounded" else 0.1,
        contradiction_prob=0.9 if label == "contradicted" else 0.05,
        neutral_prob=0.05,
        overlap_score=0.5,
        label=label,
    )


def _answer(
    query: str, *, answerable: bool, answer: str = "some answer", citations: list[Citation] | None = None
) -> GeneratedAnswer:
    return GeneratedAnswer(
        query=query,
        answer=answer,
        citations=citations if citations is not None else [],
        answerable=answerable,
        hallucinated_citation_ids=[],
        has_hallucinated_citations=False,
        model="vertex-answer-api",
        input_tokens=0,
        output_tokens=0,
        stop_reason="vertex_answer",
    )


def _verdict(verdict: str) -> JudgeVerdict:
    return JudgeVerdict(
        verdict=verdict, reasoning="because", model="claude-sonnet-5", input_tokens=5, output_tokens=5, stop_reason="tool_use"
    )


def _question(
    qid: str, *, answerable: bool, gold_chunk_ids: list[str] | None = None, gold_answer: str | None = "gold answer"
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question=qid,
        type="single_hop",
        answerable=answerable,
        gold_chunk_ids=gold_chunk_ids if gold_chunk_ids is not None else (["Test_page__000"] if answerable else []),
        gold_answer=gold_answer if answerable else None,
    )


def test_correct_abstention_needs_no_judge_call():
    q = _question("q1", answerable=False)
    retriever = FakeVertexRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=False)})
    judge = FakeJudge({})

    results = run_vertex_eval_search(
        [q], vertex_retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker()
    )

    assert results[0].correctness_label == "correct_abstention"
    assert results[0].judge is None
    assert results[0].retrieval is None
    assert judge.calls == []


def test_both_answerable_calls_judge_and_uses_its_verdict():
    q = _question("q1", answerable=True)
    retriever = FakeVertexRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True)})
    judge = FakeJudge({"q1": _verdict("correct")})

    results = run_vertex_eval_search(
        [q], vertex_retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker()
    )

    assert results[0].correctness_label == "correct"
    assert results[0].retrieval is not None
    assert results[0].retrieval.hit is True  # retriever's chunk matches gold_chunk_ids
    assert judge.calls == ["q1"]


def test_generation_failure_recorded_as_error_without_aborting_batch():
    q1 = _question("q1", answerable=True)
    q2 = _question("q2", answerable=True)
    retriever = FakeVertexRetriever({"q1": make_chunks(1), "q2": make_chunks(1)})
    generator = FakeGenerator({"q1": RuntimeError("boom"), "q2": _answer("q2", answerable=True)})
    judge = FakeJudge({"q2": _verdict("correct")})

    results = run_vertex_eval_search(
        [q1, q2], vertex_retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker()
    )

    assert results[0].correctness_label == "error"
    assert results[0].generated is None
    assert "boom" in results[0].error
    assert results[1].correctness_label == "correct"


def test_judge_failure_recorded_as_error_but_keeps_generated_answer():
    q = _question("q1", answerable=True)
    retriever = FakeVertexRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True)})
    judge = FakeJudge({"q1": RuntimeError("judge exploded")})

    results = run_vertex_eval_search(
        [q], vertex_retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker()
    )

    assert results[0].correctness_label == "error"
    assert results[0].generated is not None
    assert results[0].judge is None
    assert "judge exploded" in results[0].error


def test_grounding_computed_when_answer_has_citations():
    q = _question("q1", answerable=True)
    citation = Citation(chunk_id="Test_page__000", claim="some claim", chunk=make_chunks(1)[0])
    retriever = FakeVertexRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True, citations=[citation])})
    judge = FakeJudge({"q1": _verdict("correct")})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "grounded")])

    results = run_vertex_eval_search(
        [q], vertex_retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )

    assert results[0].grounding is not None
    assert results[0].grounding.all_grounded is True
    assert grounding_checker.calls == [[citation]]
