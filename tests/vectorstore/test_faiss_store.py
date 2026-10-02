from __future__ import annotations

import numpy as np

from rag_receipts.indexing.faiss_index import build_flat_ip_index
from rag_receipts.vectorstore.faiss_store import FaissVectorStore, dense_search

from .helpers import make_chunks_df


def test_dense_search_shapes_and_descending_order():
    embeddings = np.eye(4, dtype="float32")
    index = build_flat_ip_index(embeddings)

    scores, ids = dense_search(index, embeddings[1:2], top_k=4)

    assert scores.shape == (1, 4)
    assert ids.shape == (1, 4)
    assert ids[0][0] == 1
    assert list(scores[0]) == sorted(scores[0], reverse=True)


def test_dense_search_returns_minus_one_padding_when_top_k_exceeds_corpus():
    embeddings = np.eye(3, dtype="float32")
    index = build_flat_ip_index(embeddings)

    scores, ids = dense_search(index, embeddings[0:1], top_k=5)

    assert list(ids[0][3:]) == [-1, -1]


def test_search_joins_metadata_by_row_position():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    store = FaissVectorStore(index=build_flat_ip_index(embeddings), metadata=metadata)

    results = store.search(embeddings[1], top_k=n)

    assert results[0].chunk_id == metadata.iloc[1].chunk_id
    assert abs(results[0].score - 1.0) < 1e-5
    assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)


def test_search_filters_minus_one_padding():
    n = 3
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    store = FaissVectorStore(index=build_flat_ip_index(embeddings), metadata=metadata)

    results = store.search(embeddings[0], top_k=10)

    assert len(results) == n


def test_search_returns_empty_list_when_index_empty():
    metadata = make_chunks_df(3)
    store = FaissVectorStore(
        index=build_flat_ip_index(np.zeros((0, 4), dtype="float32")), metadata=metadata
    )

    results = store.search(np.zeros(4, dtype="float32"), top_k=3)

    assert results == []


def test_search_result_carries_full_metadata():
    n = 2
    embeddings = np.eye(n, dtype="float32")
    metadata = make_chunks_df(n)
    store = FaissVectorStore(index=build_flat_ip_index(embeddings), metadata=metadata)

    results = store.search(embeddings[0], top_k=n)

    by_id = metadata.set_index("chunk_id")
    for result in results:
        source = by_id.loc[result.chunk_id]
        assert result.page_title == source.page_title
        assert result.url == source.url
        assert result.section_path == source.section_path
        assert result.heading_anchor == source.heading_anchor
        assert result.source_type == source.source_type
        assert result.text == source.text
        assert result.token_count == source.token_count
