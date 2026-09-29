from __future__ import annotations

from rag_receipts.eval.retrieval_metrics import (
    hit,
    mrr,
    precision_at_k,
    recall_at_k,
    score_retrieval,
)
from rag_receipts.retrieval.models import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_title="Test page",
        url="https://oldschool.runescape.wiki/w/Test_page",
        section_path="",
        heading_anchor="",
        source_type="prose",
        text="text",
        token_count=10,
        dense_score=0.5,
        rerank_score=0.5,
        dense_rank=1,
        final_rank=1,
    )


def test_precision_at_k_exact_match():
    assert precision_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_precision_at_k_partial_overlap():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "z"}) == 0.25


def test_precision_at_k_no_overlap():
    assert precision_at_k(["a", "b"], {"x", "y"}) == 0.0


def test_precision_at_k_empty_retrieved():
    assert precision_at_k([], {"a"}) == 0.0


def test_recall_at_k_all_gold_found():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}) == 1.0


def test_recall_at_k_partial_recall():
    assert recall_at_k(["a", "c"], {"a", "b"}) == 0.5


def test_recall_at_k_empty_gold_set():
    assert recall_at_k(["a", "b"], set()) == 0.0


def test_mrr_first_result_is_relevant():
    assert mrr(["a", "b"], {"a"}) == 1.0


def test_mrr_second_result_is_relevant():
    assert mrr(["b", "a"], {"a"}) == 0.5


def test_mrr_no_relevant_result():
    assert mrr(["b", "c"], {"a"}) == 0.0


def test_hit_true_on_any_overlap():
    assert hit(["a", "b"], {"b", "z"}) is True


def test_hit_false_on_no_overlap():
    assert hit(["a", "b"], {"z"}) is False


def test_score_retrieval_aggregates_all_metrics():
    retrieved = [_chunk("a"), _chunk("b"), _chunk("c")]

    score = score_retrieval(retrieved, gold_chunk_ids=["b"])

    assert score.retrieved_chunk_ids == ["a", "b", "c"]
    assert score.precision == 1 / 3
    assert score.recall == 1.0
    assert score.mrr == 0.5
    assert score.hit is True


def test_score_retrieval_no_gold_match():
    retrieved = [_chunk("a"), _chunk("b")]

    score = score_retrieval(retrieved, gold_chunk_ids=["z"])

    assert score.precision == 0.0
    assert score.recall == 0.0
    assert score.mrr == 0.0
    assert score.hit is False
