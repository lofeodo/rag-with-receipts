"""Duplicated fakes for Vertex client Protocols - matches the project's
established convention (duplicated fakes rather than cross-package imports,
see tests/generation/helpers.py)."""

from __future__ import annotations

import pandas as pd

from rag_receipts.eval.models import EvalQuestion
from rag_receipts.retrieval.models import RetrievedChunk


def make_chunks(n: int = 3) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"Test_page__{i:03d}",
            page_title="Test page",
            url="https://oldschool.runescape.wiki/w/Test_page",
            section_path="Section A" if i % 2 == 0 else "",
            heading_anchor="Section_A" if i % 2 == 0 else "",
            source_type="prose",
            text=f"This is synthetic chunk number {i}.",
            token_count=50 + i,
            dense_score=0.0,
            rerank_score=0.0,
            dense_rank=i + 1,
            final_rank=i + 1,
        )
        for i in range(n)
    ]


def make_eval_questions(n: int = 3) -> list[EvalQuestion]:
    return [
        EvalQuestion(
            id=f"q{i:03d}",
            question=f"Synthetic question {i}?",
            type="single_hop",
            answerable=True,
            gold_chunk_ids=[f"Test_page__{i:03d}"],
            gold_answer=f"Synthetic gold answer {i}.",
        )
        for i in range(n)
    ]


class FakeVertexRetriever:
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


class FakeVertexAnswerer:
    def __init__(self, responses: dict[str, object]):
        self._responses = responses
        self.calls: list[str] = []

    def answer(self, query):
        self.calls.append(query)
        response = self._responses[query]
        if isinstance(response, Exception):
            raise response
        return response


class FakeGroundingChecker:
    def __init__(self, verdicts=None):
        self._verdicts = verdicts if verdicts is not None else []
        self.calls: list[list] = []

    def check(self, citations):
        self.calls.append(citations)
        if not citations:
            return []
        return self._verdicts


def make_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "Attack_range__000",
                "page_title": "Attack range",
                "url": "https://oldschool.runescape.wiki/w/Attack_range",
                "section_path": "Overview",
                "heading_anchor": "Overview",
                "token_count": 120,
                "source_type": "prose",
                "text": "Attack range is a measure of how far away...",
            },
            {
                "chunk_id": "Monkey_Madness_I__003",
                "page_title": "Monkey Madness I",
                "url": "https://oldschool.runescape.wiki/w/Monkey_Madness_I",
                "section_path": "Requirements",
                "heading_anchor": "Requirements",
                "token_count": 90,
                "source_type": "infobox",
                "text": "Items required: A gold bar, Five empty inventory slots",
            },
        ]
    )


class FakeDocument:
    def __init__(self, id: str, struct_data: dict):
        self.id = id
        self.struct_data = struct_data


class FakeSearchResult:
    def __init__(self, document: FakeDocument):
        self.document = document


class FakeSearchServiceClient:
    """Satisfies VertexSearchClientLike. Records every request it receives."""

    def __init__(self, results: list[FakeSearchResult]):
        self._results = results
        self.requests: list[object] = []

    def search(self, request):
        self.requests.append(request)
        return self._results


def make_search_result(chunk_id: str, *, document_id: str | None = None) -> FakeSearchResult:
    """document_id defaults to chunk_id (the normal case); pass a different
    value to exercise the id/structData mismatch path."""
    return FakeSearchResult(
        FakeDocument(id=document_id if document_id is not None else chunk_id, struct_data={"chunk_id": chunk_id})
    )


class FakeDocumentMetadata:
    def __init__(self, struct_data: dict | None = None, document: str | None = None):
        self.struct_data = struct_data
        self.document = document


class FakeChunkInfo:
    def __init__(self, document_metadata: FakeDocumentMetadata):
        self.document_metadata = document_metadata


class FakeReference:
    """Mirrors the real AnswerQueryResponse shape confirmed live (2026-10-02):
    chunk_id lives at reference.chunk_info.document_metadata.struct_data, not
    directly on the reference."""

    def __init__(self, struct_data: dict):
        self.chunk_info = FakeChunkInfo(FakeDocumentMetadata(struct_data=struct_data))


class FakeCitationSource:
    def __init__(self, reference_id: str):
        self.reference_id = reference_id


class FakeCitation:
    def __init__(self, sources: list[FakeCitationSource], start_index: int, end_index: int):
        self.sources = sources
        self.start_index = start_index
        self.end_index = end_index


class FakeAnswer:
    def __init__(
        self,
        *,
        answer_text: str,
        citations: list[FakeCitation] | None = None,
        references: list[FakeReference] | None = None,
        grounding_score: float | None = None,
    ):
        self.answer_text = answer_text
        self.citations = citations or []
        self.references = references or []
        self.grounding_score = grounding_score


class FakeAnswerQueryResponse:
    def __init__(self, answer: FakeAnswer):
        self.answer = answer


class FakeConversationalSearchServiceClient:
    """Satisfies VertexAnswerClientLike. Records every request it receives."""

    def __init__(self, response: FakeAnswerQueryResponse):
        self._response = response
        self.requests: list[object] = []

    def answer_query(self, request):
        self.requests.append(request)
        return self._response
