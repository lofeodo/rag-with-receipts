from __future__ import annotations

from rag_receipts.eval.models import EvalQuestion, JudgeVerdict
from rag_receipts.eval.runner import run_eval, summarize
from rag_receipts.generation.models import Citation, GeneratedAnswer
from rag_receipts.grounding.models import GroundingVerdict

from .helpers import make_chunks


class FakeGroundingChecker:
    """Structural stand-in for GroundingChecker - returns a scripted list of verdicts
    regardless of the citations passed in, unless empty citations are passed (matching
    the real GroundingChecker.check([]) == [] contract)."""

    def __init__(self, verdicts: list[GroundingVerdict] | None = None):
        self._verdicts = verdicts if verdicts is not None else []
        self.calls: list[list] = []

    def check(self, citations):
        self.calls.append(citations)
        if not citations:
            return []
        return self._verdicts


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


class FakeRetriever:
    def __init__(self, chunks_by_query: dict[str, list]):
        self._chunks_by_query = chunks_by_query

    def retrieve(self, query):
        return self._chunks_by_query.get(query, [])


class FakeGenerator:
    def __init__(self, responses: dict[str, object]):
        self._responses = responses
        self.calls: list[str] = []

    def generate(self, query, chunks):
        self.calls.append(query)
        response = self._responses[query]
        if isinstance(response, Exception):
            raise response
        return response


class FakeJudge:
    def __init__(self, responses: dict[str, object]):
        self._responses = responses
        self.calls: list[str] = []

    def score(self, question, gold_answer, generated_answer):
        self.calls.append(question)
        response = self._responses[question]
        if isinstance(response, Exception):
            raise response
        return response


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
        model="claude-sonnet-5",
        input_tokens=10,
        output_tokens=10,
        stop_reason="tool_use",
    )


def _verdict(verdict: str) -> JudgeVerdict:
    return JudgeVerdict(
        verdict=verdict,
        reasoning="because",
        model="claude-sonnet-5",
        input_tokens=5,
        output_tokens=5,
        stop_reason="tool_use",
    )


def _question(
    qid: str,
    *,
    answerable: bool,
    qtype: str = "single_hop",
    gold_chunk_ids: list[str] | None = None,
    gold_answer: str | None = "gold answer",
) -> EvalQuestion:
    return EvalQuestion(
        id=qid,
        question=qid,
        type=qtype,
        answerable=answerable,
        gold_chunk_ids=gold_chunk_ids if gold_chunk_ids is not None else (["Test_page__000"] if answerable else []),
        gold_answer=gold_answer if answerable else None,
    )


def test_correct_abstention_needs_no_judge_call():
    q = _question("q1", answerable=False)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=False)})
    judge = FakeJudge({})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "correct_abstention"
    assert results[0].judge is None
    assert results[0].retrieval is None  # answerable=False, no gold chunks to score
    assert judge.calls == []


def test_incorrectly_abstained_needs_no_judge_call():
    q = _question("q1", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=False)})
    judge = FakeJudge({})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "incorrectly_abstained"
    assert results[0].judge is None
    assert results[0].retrieval is not None
    assert judge.calls == []


def test_incorrectly_answered_needs_no_judge_call():
    q = _question("q1", answerable=False)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True)})
    judge = FakeJudge({})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "incorrectly_answered"
    assert results[0].judge is None
    assert judge.calls == []


def test_both_answerable_calls_judge_and_uses_its_verdict():
    q = _question("q1", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True)})
    judge = FakeJudge({"q1": _verdict("partially_correct")})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "partially_correct"
    assert results[0].judge is not None
    assert judge.calls == ["q1"]


def test_generation_failure_recorded_as_error_without_aborting_batch():
    q1 = _question("q1", answerable=True)
    q2 = _question("q2", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1), "q2": make_chunks(1)})
    generator = FakeGenerator(
        {"q1": RuntimeError("boom"), "q2": _answer("q2", answerable=True)}
    )
    judge = FakeJudge({"q2": _verdict("correct")})

    results = run_eval([q1, q2], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "error"
    assert results[0].generated is None
    assert "boom" in results[0].error
    assert results[1].correctness_label == "correct"


def test_judge_failure_recorded_as_error_but_keeps_generated_answer():
    q = _question("q1", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True)})
    judge = FakeJudge({"q1": RuntimeError("judge exploded")})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())

    assert results[0].correctness_label == "error"
    assert results[0].generated is not None
    assert results[0].judge is None
    assert "judge exploded" in results[0].error


def test_summarize_aggregates_retrieval_and_correctness():
    q1 = _question("q1", answerable=True, qtype="single_hop")
    q2 = _question("q2", answerable=True, qtype="multi_hop")
    q3 = _question("q3", answerable=False, qtype="single_hop")

    chunks = make_chunks(3)  # chunk_ids Test_page__000/001/002
    retriever = FakeRetriever({"q1": chunks, "q2": chunks, "q3": chunks})
    generator = FakeGenerator(
        {
            "q1": _answer("q1", answerable=True),
            "q2": _answer("q2", answerable=True),
            "q3": _answer("q3", answerable=False),
        }
    )
    judge = FakeJudge({"q1": _verdict("correct"), "q2": _verdict("incorrect")})

    q1 = _question("q1", answerable=True, qtype="single_hop", gold_chunk_ids=["Test_page__000"])
    q2 = _question("q2", answerable=True, qtype="multi_hop", gold_chunk_ids=["Test_page__999"])

    results = run_eval([q1, q2, q3], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())
    summary = summarize(results)

    assert summary.question_count == 3
    assert summary.error_count == 0
    assert summary.correctness_label_counts == {
        "correct": 1,
        "incorrect": 1,
        "correct_abstention": 1,
    }
    # 1 of 3 non-error results counted "correct" (correct_abstention counts too) -> 2/3
    assert summary.correctness_accuracy == 2 / 3
    assert summary.retrieval["count"] == 2  # only q1/q2 are answerable=True
    assert summary.retrieval_by_type["single_hop"]["count"] == 1
    assert summary.retrieval_by_type["multi_hop"]["count"] == 1
    assert summary.retrieval_by_type["single_hop"]["hit_rate"] == 1.0  # q1's gold chunk was retrieved
    assert summary.retrieval_by_type["multi_hop"]["hit_rate"] == 0.0  # q2's gold chunk wasn't
    assert summary.correctness_by_type["single_hop"]["accuracy"] == 1.0  # q1 correct, q3 correct_abstention
    assert summary.correctness_by_type["multi_hop"]["accuracy"] == 0.0  # q2 incorrect


def test_summarize_excludes_untyped_questions_from_type_breakdown():
    q1 = _question("q1", answerable=True, qtype="single_hop")
    q2 = _question("q2", answerable=False, qtype=None)  # untyped: no meaningful type

    retriever = FakeRetriever({"q1": make_chunks(1), "q2": make_chunks(1)})
    generator = FakeGenerator(
        {"q1": _answer("q1", answerable=True), "q2": _answer("q2", answerable=False)}
    )
    judge = FakeJudge({"q1": _verdict("correct")})

    results = run_eval([q1, q2], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())
    summary = summarize(results)

    # both questions still count toward the overall totals
    assert summary.question_count == 2
    assert summary.correctness_label_counts == {"correct": 1, "correct_abstention": 1}
    assert summary.correctness_accuracy == 1.0

    # but only the typed question shows up in the by-type breakdown
    assert list(summary.correctness_by_type.keys()) == ["single_hop"]
    assert summary.correctness_by_type["single_hop"]["count"] == 1
    assert list(summary.retrieval_by_type.keys()) == ["single_hop"]


def test_summarize_handles_all_errors_without_division_by_zero():
    q = _question("q1", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": RuntimeError("boom")})
    judge = FakeJudge({})

    results = run_eval([q], retriever=retriever, generator=generator, judge=judge, grounding_checker=FakeGroundingChecker())
    summary = summarize(results)

    assert summary.error_count == 1
    assert summary.correctness_accuracy == 0.0


def test_grounding_computed_when_answer_has_citations():
    q = _question("q1", answerable=True)
    citation = Citation(chunk_id="Test_page__000", claim="some claim", chunk=make_chunks(1)[0])
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True, citations=[citation])})
    judge = FakeJudge({"q1": _verdict("correct")})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "grounded")])

    results = run_eval(
        [q], retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )

    assert results[0].grounding is not None
    assert results[0].grounding.all_grounded is True
    assert grounding_checker.calls == [[citation]]


def test_grounding_is_none_when_answer_has_no_citations():
    q = _question("q1", answerable=False)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=False)})  # no citations
    judge = FakeJudge({})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "grounded")])

    results = run_eval(
        [q], retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )

    assert results[0].grounding is None


def test_grounding_is_none_on_generation_error():
    q = _question("q1", answerable=True)
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": RuntimeError("boom")})
    judge = FakeJudge({})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "grounded")])

    results = run_eval(
        [q], retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )

    assert results[0].grounding is None


def test_grounding_still_computed_when_judge_fails():
    q = _question("q1", answerable=True)
    citation = Citation(chunk_id="Test_page__000", claim="some claim", chunk=make_chunks(1)[0])
    retriever = FakeRetriever({"q1": make_chunks(1)})
    generator = FakeGenerator({"q1": _answer("q1", answerable=True, citations=[citation])})
    judge = FakeJudge({"q1": RuntimeError("judge exploded")})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "contradicted")])

    results = run_eval(
        [q], retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )

    assert results[0].correctness_label == "error"
    assert results[0].grounding is not None
    assert results[0].grounding.any_contradicted is True


def test_summarize_aggregates_grounding_stats():
    q1 = _question("q1", answerable=True, qtype="single_hop")
    q2 = _question("q2", answerable=True, qtype="multi_hop")
    q3 = _question("q3", answerable=False, qtype=None)  # no citations -> grounding None

    chunk = make_chunks(1)[0]
    citation1 = Citation(chunk_id="Test_page__000", claim="claim one", chunk=chunk)
    citation2 = Citation(chunk_id="Test_page__000", claim="claim two", chunk=chunk)

    retriever = FakeRetriever({"q1": [chunk], "q2": [chunk], "q3": [chunk]})
    generator = FakeGenerator(
        {
            "q1": _answer("q1", answerable=True, citations=[citation1]),
            "q2": _answer("q2", answerable=True, citations=[citation2]),
            "q3": _answer("q3", answerable=False),
        }
    )
    judge = FakeJudge({"q1": _verdict("correct"), "q2": _verdict("incorrect")})
    grounding_checker = FakeGroundingChecker([_grounding_verdict("Test_page__000", "ungrounded")])

    q1 = _question("q1", answerable=True, qtype="single_hop", gold_chunk_ids=["Test_page__000"])
    q2 = _question("q2", answerable=True, qtype="multi_hop", gold_chunk_ids=["Test_page__000"])

    results = run_eval(
        [q1, q2, q3], retriever=retriever, generator=generator, judge=judge, grounding_checker=grounding_checker
    )
    summary = summarize(results)

    assert summary.grounding["citation_count"] == 2
    assert summary.grounding["answer_count"] == 2
    assert summary.grounding["ungrounded_rate"] == 1.0
    assert summary.grounding["fully_grounded_answer_rate"] == 0.0
    assert summary.grounding_by_type["single_hop"]["citation_count"] == 1
    assert summary.grounding_by_type["multi_hop"]["citation_count"] == 1
    assert summary.grounding_by_correctness_label["correct"]["citation_count"] == 1
    assert summary.grounding_by_correctness_label["incorrect"]["citation_count"] == 1
    assert summary.grounding_by_correctness_label["correct_abstention"]["citation_count"] == 0
