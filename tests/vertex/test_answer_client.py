import pytest

from rag_receipts.vertex.answer_client import VertexAnswerer, _extract_reference_chunk_id
from rag_receipts.vertex.config import VertexConfig

from .helpers import (
    FakeAnswer,
    FakeAnswerQueryResponse,
    FakeCitation,
    FakeCitationSource,
    FakeConversationalSearchServiceClient,
    FakeDocumentMetadata,
    FakeReference,
    make_metadata_df,
)


def _make_answerer(client, metadata=None):
    return VertexAnswerer(
        config=VertexConfig(),
        client=client,
        metadata=metadata if metadata is not None else make_metadata_df(),
        request_factory=lambda query: {"query": query},
    )


def test_answer_returns_generated_answer_with_resolved_citation():
    answer_text = "Attack range is a measure of distance."
    response = FakeAnswerQueryResponse(
        FakeAnswer(
            answer_text=answer_text,
            citations=[FakeCitation(sources=[FakeCitationSource(reference_id="0")], start_index=0, end_index=len(answer_text))],
            references=[FakeReference(struct_data={"chunk_id": "Attack_range__000"})],
            grounding_score=0.87,
        )
    )
    client = FakeConversationalSearchServiceClient(response)
    answerer = _make_answerer(client)

    generated, grounding_score = answerer.answer("what is attack range")

    assert generated.answer == answer_text
    assert generated.answerable is True
    assert len(generated.citations) == 1
    assert generated.citations[0].chunk_id == "Attack_range__000"
    assert generated.citations[0].chunk.page_title == "Attack range"
    assert generated.citations[0].claim == answer_text
    assert generated.hallucinated_citation_ids == []
    assert generated.has_hallucinated_citations is False
    assert generated.model == "vertex-answer-api"
    assert generated.input_tokens == 0
    assert generated.output_tokens == 0
    assert grounding_score == 0.87
    assert client.requests == [{"query": "what is attack range"}]


def test_answer_unresolvable_reference_chunk_id_recorded_as_hallucinated():
    response = FakeAnswerQueryResponse(
        FakeAnswer(
            answer_text="Some answer.",
            citations=[FakeCitation(sources=[FakeCitationSource(reference_id="0")], start_index=0, end_index=12)],
            references=[FakeReference(struct_data={"chunk_id": "Nonexistent_chunk__999"})],
        )
    )
    client = FakeConversationalSearchServiceClient(response)
    answerer = _make_answerer(client)

    generated, _ = answerer.answer("some query")

    assert generated.citations == []
    assert generated.hallucinated_citation_ids == ["Nonexistent_chunk__999"]
    assert generated.has_hallucinated_citations is True


def test_answer_empty_answer_text_is_not_answerable():
    response = FakeAnswerQueryResponse(FakeAnswer(answer_text=""))
    client = FakeConversationalSearchServiceClient(response)
    answerer = _make_answerer(client)

    generated, _ = answerer.answer("out of corpus query")

    assert generated.answerable is False
    assert generated.citations == []


def test_answer_missing_grounding_score_returns_none():
    response = FakeAnswerQueryResponse(FakeAnswer(answer_text="Some answer."))
    client = FakeConversationalSearchServiceClient(response)
    answerer = _make_answerer(client)

    _, grounding_score = answerer.answer("some query")

    assert grounding_score is None


def test_answer_missing_answer_field_raises_runtime_error():
    class BrokenResponse:
        pass

    client = FakeConversationalSearchServiceClient(BrokenResponse())
    answerer = _make_answerer(client)

    with pytest.raises(RuntimeError, match="no 'answer' field"):
        answerer.answer("some query")


def test_extract_reference_chunk_id_falls_back_to_document_metadata_without_chunk_info():
    """A reference with no chunk_info at all (e.g. a plain document reference)
    should still resolve via the document_metadata/document-resource-name
    fallback path, not just the chunk_info.document_metadata primary path."""

    class BareReference:
        def __init__(self, document_metadata):
            self.document_metadata = document_metadata

    reference = BareReference(FakeDocumentMetadata(document="projects/p/.../documents/Attack_range__004"))

    assert _extract_reference_chunk_id(reference) == "Attack_range__004"


def test_answer_no_citations_returns_empty_citations_not_error():
    response = FakeAnswerQueryResponse(FakeAnswer(answer_text="An answer with no citations."))
    client = FakeConversationalSearchServiceClient(response)
    answerer = _make_answerer(client)

    generated, _ = answerer.answer("some query")

    assert generated.citations == []
    assert generated.hallucinated_citation_ids == []
