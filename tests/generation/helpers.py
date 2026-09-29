from __future__ import annotations

from rag_receipts.retrieval.models import RetrievedChunk


class FakeUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, input: dict):
        self.input = input


class FakeMessage:
    """Deterministic fake standing in for anthropic.types.Message - only the
    attributes Generator.generate reads (content, stop_reason, usage)."""

    def __init__(
        self,
        *,
        content: list,
        stop_reason: str,
        input_tokens: int = 10,
        output_tokens: int = 20,
    ):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeMessagesResource:
    def __init__(self, responses: list[FakeMessage]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeAnthropicClient:
    """Structural stand-in for AnthropicClientLike - a .messages attribute with
    .create(...) returning scripted FakeMessage responses in order."""

    def __init__(self, responses: list[FakeMessage]):
        self.messages = FakeMessagesResource(responses)


def make_tool_response(
    *,
    answerable: bool,
    answer: str,
    citations: list[dict],
    stop_reason: str = "tool_use",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> FakeMessage:
    payload = {"answerable": answerable, "answer": answer, "citations": citations}
    return FakeMessage(
        content=[FakeToolUseBlock(payload)],
        stop_reason=stop_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


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
            dense_score=1.0 - i * 0.1,
            rerank_score=1.0 - i * 0.1,
            dense_rank=i + 1,
            final_rank=i + 1,
        )
        for i in range(n)
    ]
