"""Fakes for tests/api - structural RetrieverLike/GeneratorLike implementations,
same pattern as tests/generation/helpers.py::FakeAnthropicClient (a fake
satisfying the Protocol, not a mock of internals)."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_receipts.generation.models import Citation, GeneratedAnswer
from rag_receipts.retrieval.models import RetrievedChunk
from rag_receipts.telemetry.models import RetrievalTiming


def make_chunk(chunk_id: str = "chunk_001") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_title="Monkey Madness I",
        url="https://oldschool.runescape.wiki/w/Monkey_Madness_I",
        section_path="Monkey Madness I > Requirements",
        heading_anchor="Requirements",
        source_type="prose",
        text="A gold bar and five empty inventory slots are required.",
        token_count=12,
        dense_score=0.9,
        rerank_score=0.99,
        dense_rank=1,
        final_rank=1,
    )


@dataclass
class FakeRetriever:
    chunks: list[RetrievedChunk] = field(default_factory=lambda: [make_chunk()])
    timing: RetrievalTiming = field(
        default_factory=lambda: RetrievalTiming(
            embed_query_s=0.01, dense_search_s=0.001, rerank_s=0.02, total_s=0.031
        )
    )

    def retrieve_with_timing(self, query: str) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        return self.chunks, self.timing


@dataclass
class FakeGenerator:
    answer: GeneratedAnswer | None = None

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if self.answer is not None:
            return self.answer
        citations = (
            [Citation(chunk_id=chunks[0].chunk_id, claim="A gold bar is required.", chunk=chunks[0])]
            if chunks
            else []
        )
        return GeneratedAnswer(
            query=query,
            answer="A gold bar and five empty inventory slots are required.",
            citations=citations,
            answerable=True,
            hallucinated_citation_ids=[],
            has_hallucinated_citations=False,
            model="claude-sonnet-5",
            input_tokens=100,
            output_tokens=50,
            stop_reason="tool_use",
        )


class RaisingGenerator:
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        raise RuntimeError("stop_reason=max_tokens - possibly truncated tool call")
