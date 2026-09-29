from __future__ import annotations

from rag_receipts.eval.models import EvalQuestion
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
    attributes Judge.score reads (content, stop_reason, usage)."""

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


def make_verdict_response(
    *,
    verdict: str,
    reasoning: str = "reasoning",
    stop_reason: str = "tool_use",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> FakeMessage:
    payload = {"verdict": verdict, "reasoning": reasoning}
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


def make_eval_questions(n: int = 3) -> list[EvalQuestion]:
    """n answerable single-hop questions, each with one gold chunk_id matching
    make_chunks()'s naming scheme."""
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
