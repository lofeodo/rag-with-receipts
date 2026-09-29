"""Claude Sonnet 5 generation with structured, chunk-validated citations (Step 4).

Every citation the model returns is untrusted input: a chunk_id it names could be
one it invented rather than one actually in the retrieved set. generate() resolves
every citation against the chunks it was given and separates the two outcomes
(Citation vs hallucinated_citation_ids) instead of rendering citations on faith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic

from rag_receipts.generation.config import GenerationConfig
from rag_receipts.generation.models import Citation, GeneratedAnswer
from rag_receipts.ingestion.utils import breadcrumb
from rag_receipts.retrieval.models import RetrievedChunk

ANSWER_TOOL_NAME = "submit_answer"

SYSTEM_PROMPT = """You are a question-answering assistant for the Old School RuneScape
Wiki. You will be given a user question and a numbered set of retrieved wiki chunks.

Rules:
- Answer using ONLY information present in the retrieved chunks below. Do not use
  outside knowledge of OSRS, even if you believe it is correct.
- If the retrieved chunks do not contain enough information to answer the question,
  set answerable to false and briefly explain what's missing in `answer` - do not guess.
- Every factual claim in your answer must be backed by at least one citation naming
  the exact chunk_id (copied verbatim) of a chunk that supports it.
- Always respond by calling the submit_answer tool - never respond with plain text."""

ANSWER_TOOL = {
    "name": ANSWER_TOOL_NAME,
    "description": (
        "Submit the final grounded answer, citing the specific retrieved chunk_ids "
        "that support each claim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answerable": {
                "type": "boolean",
                "description": "True if the retrieved chunks contain enough information to answer.",
            },
            "answer": {
                "type": "string",
                "description": (
                    "The answer, using ONLY the retrieved chunks. If answerable is "
                    "false, a brief explanation of what's missing instead."
                ),
            },
            "citations": {
                "type": "array",
                "description": "One entry per claim. Empty if answerable is false.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "Exact chunk_id copied verbatim from the context below.",
                        },
                        "claim": {
                            "type": "string",
                            "description": "The specific claim this chunk supports.",
                        },
                    },
                    "required": ["chunk_id", "claim"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answerable", "answer", "citations"],
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
    """Not lru_cache'd - deliberate deviation from Embedder/Reranker's loader
    pattern. Unlike a SentenceTransformer/CrossEncoder, constructing the Anthropic
    client is cheap (no weights to load), so a fresh client per Generator costs
    nothing. Reads ANTHROPIC_API_KEY (or other SDK-recognized credential sources)
    from the environment - never pass a key from config.yaml."""
    return anthropic.Anthropic()


def format_chunk(chunk: RetrievedChunk) -> str:
    return f"[chunk_id: {chunk.chunk_id}]\n{breadcrumb(chunk.page_title, chunk.section_path)}\n{chunk.text}"


def build_user_message(query: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(format_chunk(c) for c in chunks)
    return f"Question: {query}\n\nRetrieved context (cite chunk_id exactly as shown):\n\n{context}"


@dataclass
class Generator:
    config: GenerationConfig
    client: AnthropicClientLike

    @classmethod
    def from_config(cls, config: GenerationConfig) -> "Generator":
        return cls(config=config, client=build_anthropic_client())

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        chunk_lookup = {c.chunk_id: c for c in chunks}
        # No `temperature` - Sonnet 5 runs adaptive thinking by default, and sampling
        # params 400 whenever thinking is active. Forced tool_choice is valid on
        # claude-sonnet-5 today, but 400s on Fable 5.1/Mythos 5.1/Opus 5.5/Sonnet 5.5 -
        # if config.model is ever pointed at one of those, switch to
        # tool_choice={"type": "auto"} + a prompt instruction naming the tool.
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(query, chunks)}],
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": ANSWER_TOOL_NAME},
        )

        if response.stop_reason == "refusal":
            return GeneratedAnswer(
                query=query,
                answer="[refused]",
                citations=[],
                answerable=False,
                hallucinated_citation_ids=[],
                has_hallucinated_citations=False,
                model=self.config.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                stop_reason=response.stop_reason,
            )

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Hit max_tokens ({self.config.max_tokens}) before completing the "
                f"tool call - raise generation.max_tokens in config.yaml."
            )

        tool_use = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise RuntimeError(
                f"Expected a tool_use block from {ANSWER_TOOL_NAME}, got stop_reason="
                f"{response.stop_reason!r} with no tool call - {response.content!r}"
            )

        payload = tool_use.input  # SDK returns tool input pre-parsed as a dict
        citations: list[Citation] = []
        hallucinated_ids: list[str] = []
        for raw in payload.get("citations", []):
            chunk = chunk_lookup.get(raw["chunk_id"])
            if chunk is None:
                hallucinated_ids.append(raw["chunk_id"])
                continue
            citations.append(Citation(chunk_id=raw["chunk_id"], claim=raw["claim"], chunk=chunk))

        return GeneratedAnswer(
            query=query,
            answer=payload["answer"],
            citations=citations,
            answerable=payload["answerable"],
            hallucinated_citation_ids=hallucinated_ids,
            has_hallucinated_citations=bool(hallucinated_ids),
            model=self.config.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )
