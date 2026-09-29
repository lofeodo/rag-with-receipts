"""LLM-as-judge correctness scoring (Step 5): grades a generated answer against a gold
reference answer on a 3-way categorical verdict.

Scores correctness only, not groundedness - whether the answer is faithfully supported
by its cited chunks is Step 6's entailment/overlap check, not this judge's job. Mirrors
generation/generator.py's Protocol + @dataclass + from_config shape and stop_reason
handling discipline, with a simpler forced tool schema (no citation resolution needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from rag_receipts.eval.config import EvalConfig
from rag_receipts.eval.models import JudgeVerdict

VERDICT_TOOL_NAME = "submit_verdict"

SYSTEM_PROMPT = """You are grading a question-answering system for the Old School RuneScape
Wiki. You will be given a question, a gold reference answer, and a candidate answer
produced by the system. Judge only whether the candidate answer is factually correct
relative to the gold reference answer - do not penalize differences in phrasing, length,
or extra correct detail not present in the reference.

Verdicts:
- "correct": the candidate answer's key facts match the gold reference answer.
- "partially_correct": the candidate answer gets some key facts right but misses or
  garbles others (common on multi-hop questions requiring several facts synthesized
  together).
- "incorrect": the candidate answer's key facts contradict or are unrelated to the gold
  reference answer.

Always respond by calling the submit_verdict tool - never respond with plain text."""

VERDICT_TOOL = {
    "name": VERDICT_TOOL_NAME,
    "description": "Submit the correctness verdict for the candidate answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["correct", "partially_correct", "incorrect"],
                "description": "The correctness verdict.",
            },
            "reasoning": {
                "type": "string",
                "description": "A short explanation for the verdict.",
            },
        },
        "required": ["verdict", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}


class MessagesResource(Protocol):
    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict,
    ) -> Any: ...


class AnthropicClientLike(Protocol):
    messages: MessagesResource


def build_anthropic_client() -> anthropic.Anthropic:
    """Not lru_cache'd, matching Generator's rationale - constructing the client is
    cheap. Reads ANTHROPIC_API_KEY from the environment, never from config.yaml."""
    return anthropic.Anthropic()


def build_judge_message(question: str, gold_answer: str, generated_answer: str) -> str:
    return (
        f"Question: {question}\n\n"
        f"Gold reference answer: {gold_answer}\n\n"
        f"Candidate answer: {generated_answer}"
    )


@dataclass
class Judge:
    config: EvalConfig
    client: AnthropicClientLike

    @classmethod
    def from_config(cls, config: EvalConfig) -> "Judge":
        return cls(config=config, client=build_anthropic_client())

    def score(self, question: str, gold_answer: str, generated_answer: str) -> JudgeVerdict:
        # No `temperature` - same adaptive-thinking-vs-forced-tool_choice constraint as
        # Generator. Forced tool_choice is valid on claude-sonnet-5 today, but 400s on
        # Fable 5.1/Mythos 5.1/Opus 5.5/Sonnet 5.5 - if config.judge_model is ever pointed
        # at one of those, switch to tool_choice={"type": "auto"} + a prompt instruction.
        response = self.client.messages.create(
            model=self.config.judge_model,
            max_tokens=self.config.judge_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_judge_message(question, gold_answer, generated_answer),
                }
            ],
            tools=[VERDICT_TOOL],
            tool_choice={"type": "tool", "name": VERDICT_TOOL_NAME},
        )

        if response.stop_reason == "refusal":
            return JudgeVerdict(
                verdict="incorrect",
                reasoning="Judge model refused to score this answer.",
                model=self.config.judge_model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            )

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Hit max_tokens ({self.config.judge_max_tokens}) before completing the "
                f"tool call - raise eval.judge_max_tokens in config.yaml."
            )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError(
                f"Expected a tool_use block from {VERDICT_TOOL_NAME}, got stop_reason="
                f"{response.stop_reason!r} with no tool call - {response.content!r}"
            )

        payload = tool_use.input
        return JudgeVerdict(
            verdict=payload["verdict"],
            reasoning=payload["reasoning"],
            model=self.config.judge_model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
