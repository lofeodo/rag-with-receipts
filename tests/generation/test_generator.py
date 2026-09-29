from __future__ import annotations

import pytest

from rag_receipts.generation.config import GenerationConfig
from rag_receipts.generation.generator import Generator, build_user_message

from .helpers import FakeAnthropicClient, make_chunks, make_tool_response


def _generator(client: FakeAnthropicClient) -> Generator:
    return Generator(config=GenerationConfig(), client=client)


def test_build_user_message_includes_every_chunk_text_and_chunk_id():
    chunks = make_chunks(3)
    message = build_user_message("some query", chunks)

    for chunk in chunks:
        assert chunk.chunk_id in message
        assert chunk.text in message
    assert "some query" in message


def test_generate_answerable_false_passes_through_untouched():
    chunks = make_chunks(2)
    client = FakeAnthropicClient(
        [make_tool_response(answerable=False, answer="Not enough info", citations=[])]
    )

    result = _generator(client).generate("some query", chunks)

    assert result.answerable is False
    assert result.answer == "Not enough info"
    assert result.citations == []
    assert result.hallucinated_citation_ids == []
    assert result.has_hallucinated_citations is False


def test_generate_resolves_real_citation_to_full_chunk():
    chunks = make_chunks(3)
    target = chunks[1]
    client = FakeAnthropicClient(
        [
            make_tool_response(
                answerable=True,
                answer="The answer",
                citations=[{"chunk_id": target.chunk_id, "claim": "supports the answer"}],
            )
        ]
    )

    result = _generator(client).generate("some query", chunks)

    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == target.chunk_id
    assert result.citations[0].chunk == target
    assert result.citations[0].claim == "supports the answer"
    assert result.hallucinated_citation_ids == []
    assert result.has_hallucinated_citations is False


def test_generate_drops_hallucinated_citation_and_tracks_it():
    chunks = make_chunks(2)
    client = FakeAnthropicClient(
        [
            make_tool_response(
                answerable=True,
                answer="The answer",
                citations=[
                    {"chunk_id": chunks[0].chunk_id, "claim": "real citation"},
                    {"chunk_id": "not_a_real_chunk_id", "claim": "made up citation"},
                ],
            )
        ]
    )

    result = _generator(client).generate("some query", chunks)

    assert [c.chunk_id for c in result.citations] == [chunks[0].chunk_id]
    assert result.hallucinated_citation_ids == ["not_a_real_chunk_id"]
    assert result.has_hallucinated_citations is True


def test_generate_captures_token_usage():
    chunks = make_chunks(1)
    client = FakeAnthropicClient(
        [
            make_tool_response(
                answerable=True,
                answer="answer",
                citations=[],
                input_tokens=123,
                output_tokens=45,
            )
        ]
    )

    result = _generator(client).generate("some query", chunks)

    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert result.model == GenerationConfig().model
    assert result.stop_reason == "tool_use"


def test_generate_refusal_returns_safe_result_instead_of_raising():
    from .helpers import FakeMessage

    chunks = make_chunks(1)
    client = FakeAnthropicClient(
        [FakeMessage(content=[], stop_reason="refusal", input_tokens=5, output_tokens=1)]
    )

    result = _generator(client).generate("some query", chunks)

    assert result.answerable is False
    assert result.citations == []
    assert result.stop_reason == "refusal"


def test_generate_max_tokens_raises_runtime_error():
    from .helpers import FakeMessage

    chunks = make_chunks(1)
    client = FakeAnthropicClient(
        [FakeMessage(content=[], stop_reason="max_tokens", input_tokens=5, output_tokens=4096)]
    )

    with pytest.raises(RuntimeError, match="max_tokens"):
        _generator(client).generate("some query", chunks)


def test_generate_missing_tool_use_block_raises_runtime_error():
    from .helpers import FakeMessage

    class TextBlock:
        type = "text"
        text = "I decided to just talk instead of calling the tool."

    chunks = make_chunks(1)
    client = FakeAnthropicClient(
        [FakeMessage(content=[TextBlock()], stop_reason="end_turn", input_tokens=5, output_tokens=10)]
    )

    with pytest.raises(RuntimeError, match="tool_use"):
        _generator(client).generate("some query", chunks)
