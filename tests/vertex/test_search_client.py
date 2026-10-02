from rag_receipts.vertex.config import VertexConfig
from rag_receipts.vertex.search_client import VertexRetriever, parse_search_results

from .helpers import FakeSearchServiceClient, make_metadata_df, make_search_result


def test_parse_search_results_order_preserved():
    metadata = make_metadata_df()
    results = [
        make_search_result("Monkey_Madness_I__003"),
        make_search_result("Attack_range__000"),
    ]

    chunks = parse_search_results(results, metadata, top_k=10)

    assert [c.chunk_id for c in chunks] == ["Monkey_Madness_I__003", "Attack_range__000"]
    assert [c.dense_rank for c in chunks] == [1, 2]
    assert [c.final_rank for c in chunks] == [1, 2]


def test_parse_search_results_unresolvable_chunk_id_dropped():
    metadata = make_metadata_df()
    results = [
        make_search_result("Attack_range__000"),
        make_search_result("Nonexistent_chunk__999"),
    ]

    chunks = parse_search_results(results, metadata, top_k=10)

    assert [c.chunk_id for c in chunks] == ["Attack_range__000"]


def test_parse_search_results_respects_top_k():
    metadata = make_metadata_df()
    results = [
        make_search_result("Attack_range__000"),
        make_search_result("Monkey_Madness_I__003"),
    ]

    chunks = parse_search_results(results, metadata, top_k=1)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "Attack_range__000"


def test_parse_search_results_placeholder_scores():
    metadata = make_metadata_df()
    results = [make_search_result("Attack_range__000")]

    [chunk] = parse_search_results(results, metadata, top_k=10)

    assert chunk.dense_score == 0.0
    assert chunk.rerank_score == 0.0


def test_vertex_retriever_retrieve_returns_real_retrieved_chunks():
    metadata = make_metadata_df()
    client = FakeSearchServiceClient([make_search_result("Attack_range__000")])
    retriever = VertexRetriever(
        config=VertexConfig(search_top_k=5),
        client=client,
        metadata=metadata,
        request_factory=lambda query: {"query": query},
    )

    chunks = retriever.retrieve("what is attack range")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "Attack_range__000"
    assert chunks[0].page_title == "Attack range"
    assert client.requests == [{"query": "what is attack range"}]


def test_vertex_retriever_id_structdata_mismatch_prefers_struct_data():
    metadata = make_metadata_df()
    client = FakeSearchServiceClient(
        [make_search_result("Attack_range__000", document_id="some-other-id")]
    )
    retriever = VertexRetriever(
        config=VertexConfig(search_top_k=5),
        client=client,
        metadata=metadata,
        request_factory=lambda query: {"query": query},
    )

    chunks = retriever.retrieve("what is attack range")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "Attack_range__000"
