from __future__ import annotations

import pytest

from rag_receipts.eval.config import EvalConfig
from rag_receipts.eval.judge import Judge, build_judge_message

from .helpers import FakeAnthropicClient, FakeMessage, make_verdict_response


def _judge(client: FakeAnthropicClient) -> Judge:
    return Judge(config=EvalConfig(), client=client)


def test_build_judge_message_includes_question_gold_and_candidate():
    message = build_judge_message("some question", "gold answer text", "candidate answer text")

    assert "some question" in message
    assert "gold answer text" in message
    assert "candidate answer text" in message


def test_score_resolves_correct_verdict():
    client = FakeAnthropicClient([make_verdict_response(verdict="correct", reasoning="matches")])

    result = _judge(client).score("q", "gold", "candidate")

    assert result.verdict == "correct"
    assert result.reasoning == "matches"
    assert result.model == EvalConfig().judge_model
    assert result.stop_reason == "tool_use"


def test_score_resolves_partially_correct_verdict():
    client = FakeAnthropicClient(
        [make_verdict_response(verdict="partially_correct", reasoning="missed one fact")]
    )

    result = _judge(client).score("q", "gold", "candidate")

    assert result.verdict == "partially_correct"


def test_score_resolves_incorrect_verdict():
    client = FakeAnthropicClient([make_verdict_response(verdict="incorrect", reasoning="wrong")])

    result = _judge(client).score("q", "gold", "candidate")

    assert result.verdict == "incorrect"


def test_score_captures_token_usage():
    client = FakeAnthropicClient(
        [make_verdict_response(verdict="correct", input_tokens=55, output_tokens=12)]
    )

    result = _judge(client).score("q", "gold", "candidate")

    assert result.input_tokens == 55
    assert result.output_tokens == 12


def test_score_refusal_returns_safe_incorrect_verdict_instead_of_raising():
    client = FakeAnthropicClient(
        [FakeMessage(content=[], stop_reason="refusal", input_tokens=5, output_tokens=1)]
    )

    result = _judge(client).score("q", "gold", "candidate")

    assert result.verdict == "incorrect"
    assert result.stop_reason == "refusal"


def test_score_max_tokens_raises_runtime_error():
    client = FakeAnthropicClient(
        [FakeMessage(content=[], stop_reason="max_tokens", input_tokens=5, output_tokens=1024)]
    )

    with pytest.raises(RuntimeError, match="max_tokens"):
        _judge(client).score("q", "gold", "candidate")


def test_score_missing_tool_use_block_raises_runtime_error():
    class TextBlock:
        type = "text"
        text = "I decided to just talk instead of calling the tool."

    client = FakeAnthropicClient(
        [FakeMessage(content=[TextBlock()], stop_reason="end_turn", input_tokens=5, output_tokens=10)]
    )

    with pytest.raises(RuntimeError, match="tool_use"):
        _judge(client).score("q", "gold", "candidate")
