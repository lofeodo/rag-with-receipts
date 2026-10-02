from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .helpers import FakeCrossEncoder, FakeEncoder, make_chunks_df

from rag_receipts.indexing.config import EmbeddingConfig
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.indexing.faiss_index import build_flat_ip_index
from rag_receipts.retrieval.config import RerankerConfig, RetrievalConfig
from rag_receipts.retrieval.pipeline import Retriever, run_retrieval, run_retrieval_with_timing
from rag_receipts.retrieval.reranker import Reranker
from rag_receipts.vectorstore.faiss_store import FaissVectorStore


class ScriptedEncoder:
    """Fake encoder that ignores its input text and always returns a fixed vector -
    used where a test needs full control over the query embedding, decoupled from
    whatever text happens to be passed."""

    def __init__(self, vector):
        self.vector = vector

    def encode(self, sentences, *, batch_size, normalize_embeddings, show_progress_bar):
        return np.array([self.vector for _ in sentences], dtype="float32")


def _reranker():
    return Reranker(config=RerankerConfig(), model=FakeCrossEncoder())


def _embedder(model=None):
    return Embedder(config=EmbeddingConfig(), model=model or FakeEncoder())


def _store(embeddings: np.ndarray, metadata: pd.DataFrame) -> FaissVectorStore:
    return FaissVectorStore(index=build_flat_ip_index(embeddings), metadata=metadata)


def test_run_retrieval_filters_minus_one_ids_without_error():
    embeddings = np.eye(3, dtype="float32")
    metadata = make_chunks_df(3)

    result = run_retrieval(
        "some query",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=10, top_k_final=5),
    )

    assert len(result) <= 3


def test_run_retrieval_sorts_by_rerank_score_not_dense_score():
    # A is closer to the query vector (higher dense score) but shares no words
    # with the query text; B is further in vector space but shares every word.
    embeddings = np.array([[1.0, 0.0], [0.9, np.sqrt(1 - 0.9**2)]], dtype="float32")
    metadata = pd.DataFrame(
        [
            {
                "chunk_id": "A",
                "page_title": "Page A",
                "url": "https://example.invalid/A",
                "section_path": "",
                "heading_anchor": "",
                "token_count": 10,
                "source_type": "prose",
                "text": "unrelated filler content",
            },
            {
                "chunk_id": "B",
                "page_title": "Page B",
                "url": "https://example.invalid/B",
                "section_path": "",
                "heading_anchor": "",
                "token_count": 10,
                "source_type": "prose",
                "text": "abyssal whip special attack",
            },
        ]
    )

    result = run_retrieval(
        "abyssal whip special attack",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=2, top_k_final=2),
    )

    assert [r.chunk_id for r in result] == ["B", "A"]
    assert result[0].dense_rank == 2  # B was ranked second by dense search
    assert result[1].dense_rank == 1  # A was ranked first by dense search
    assert [r.final_rank for r in result] == [1, 2]


def test_run_retrieval_truncates_to_top_k_final_and_keeps_original_dense_rank():
    n = 5
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)

    result = run_retrieval(
        "zzz nonexistent query words",  # shares no words with any chunk text -> rerank ties
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=n, top_k_final=2),
    )

    assert len(result) == 2
    # rerank scores all tie at 0 -> stable sort preserves dense order
    assert [r.dense_rank for r in result] == [1, 2]
    assert [r.final_rank for r in result] == [1, 2]


def test_run_retrieval_final_rank_is_1_indexed_contiguous():
    n = 5
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)

    result = run_retrieval(
        "zzz nonexistent query words",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=n, top_k_final=n),
    )

    assert [r.final_rank for r in result] == [1, 2, 3, 4, 5]


def test_run_retrieval_carries_citation_fields():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)

    result = run_retrieval(
        "some query",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=n, top_k_final=n),
    )

    by_id = metadata.set_index("chunk_id")
    for chunk in result:
        source = by_id.loc[chunk.chunk_id]
        assert chunk.page_title == source.page_title
        assert chunk.url == source.url
        assert chunk.section_path == source.section_path
        assert chunk.heading_anchor == source.heading_anchor
        assert chunk.source_type == source.source_type
        assert chunk.text == source.text
        assert chunk.token_count == source.token_count


def test_run_retrieval_empty_result_when_no_valid_ids():
    metadata = make_chunks_df(3)
    store = _store(np.zeros((0, 4), dtype="float32"), metadata)

    result = run_retrieval(
        "some query",
        vector_store=store,
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=3, top_k_final=3),
    )

    assert result == []


def test_run_retrieval_with_timing_returns_same_chunks_as_run_retrieval():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    config = RetrievalConfig(top_k_dense=n, top_k_final=n)

    chunks, timing = run_retrieval_with_timing(
        "some query",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=config,
    )
    from_run_retrieval = run_retrieval(
        "some query",
        vector_store=_store(embeddings, metadata),
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=config,
    )

    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in from_run_retrieval]
    assert timing.embed_query_s >= 0.0
    assert timing.dense_search_s >= 0.0
    assert timing.rerank_s >= 0.0
    assert timing.total_s == pytest.approx(
        timing.embed_query_s + timing.dense_search_s + timing.rerank_s
    )


def test_run_retrieval_with_timing_empty_result_has_zero_rerank_time():
    metadata = make_chunks_df(3)
    store = _store(np.zeros((0, 4), dtype="float32"), metadata)

    chunks, timing = run_retrieval_with_timing(
        "some query",
        vector_store=store,
        embedder=_embedder(ScriptedEncoder([1.0, 0.0, 0.0, 0.0])),
        reranker=_reranker(),
        config=RetrievalConfig(top_k_dense=3, top_k_final=3),
    )

    assert chunks == []
    assert timing.rerank_s == 0.0
    assert timing.total_s == pytest.approx(timing.embed_query_s + timing.dense_search_s)


def test_retriever_retrieve_with_timing_delegates_to_function():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    embedder = _embedder(FakeEncoder(dim=n))
    reranker = _reranker()
    config = RetrievalConfig(top_k_dense=n, top_k_final=2)

    retriever = Retriever(
        config=config,
        vector_store=_store(embeddings, metadata),
        embedder=embedder,
        reranker=reranker,
    )

    chunks, timing = retriever.retrieve_with_timing("some query")

    assert len(chunks) == 2
    assert timing.total_s >= 0.0


def test_retriever_retrieve_delegates_to_run_retrieval():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    embedder = _embedder(FakeEncoder(dim=n))
    reranker = _reranker()
    config = RetrievalConfig(top_k_dense=n, top_k_final=2)
    store = _store(embeddings, metadata)

    retriever = Retriever(config=config, vector_store=store, embedder=embedder, reranker=reranker)

    from_retriever = retriever.retrieve("some query")
    from_function = run_retrieval(
        "some query", vector_store=store, embedder=embedder, reranker=reranker, config=config
    )

    assert [r.chunk_id for r in from_retriever] == [r.chunk_id for r in from_function]
    assert [r.rerank_score for r in from_retriever] == [r.rerank_score for r in from_function]
