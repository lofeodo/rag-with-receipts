from dataclasses import dataclass

import numpy as np

from rag_receipts.vectorstore.base import ScoredChunk, VectorStore


def test_scored_chunk_fields():
    chunk = ScoredChunk(
        chunk_id="c1",
        page_title="Page",
        url="https://example.com",
        section_path="Section > Sub",
        heading_anchor="sub",
        source_type="prose",
        text="some text",
        token_count=42,
        score=0.9876,
    )

    assert chunk.chunk_id == "c1"
    assert chunk.score == 0.9876
    assert chunk.token_count == 42


def test_minimal_fake_satisfies_vector_store_protocol():
    """Structural typing check: a minimal class implementing .search() with the
    right signature satisfies VectorStore without inheriting from it."""

    @dataclass
    class FakeStore:
        def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
            return []

    store: VectorStore = FakeStore()
    assert store.search(np.zeros(4, dtype="float32"), 5) == []
