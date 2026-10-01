"""Config B: Vertex AI Search's own Answer API (retrieval + generation +
Vertex's own citations) end-to-end - the "fully managed alternative" half of
the Vertex AI Search comparison stretch goal.

The exact AnswerQueryResponse field paths used below (answer.citations,
answer.references, answer.grounding_score) are sourced from Discovery Engine's
public API docs and are NOT yet confirmed against a live response - the API
has had v1/v1beta/v1alpha field differences historically. CLAUDE.md's Vertex
stretch-goal status note requires printing one raw response and confirming
these paths before trusting a full batch run (see scripts/run_vertex_eval.py's
--mode answer).

Citations are resolved against this project's own index_metadata.parquet via
resolve_chunk() rather than trusted from whatever inline text Vertex itself
returns - sidesteps uncertainty about Vertex's returned passage text entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from rag_receipts.generation.models import Citation, GeneratedAnswer
from rag_receipts.retrieval.pipeline import load_metadata
from rag_receipts.vertex.config import VertexConfig
from rag_receipts.vertex.resolve import resolve_chunk, struct_to_dict


class VertexAnswerClientLike(Protocol):
    def answer_query(self, request: Any) -> Any: ...


def _build_answer_request(query: str, config: VertexConfig) -> Any:
    from google.cloud import discoveryengine_v1 as discoveryengine

    serving_config = (
        f"projects/{config.project_id}/locations/{config.location}/"
        f"collections/default_collection/engines/{config.engine_id}/"
        f"servingConfigs/{config.answer_serving_config}"
    )
    return discoveryengine.AnswerQueryRequest(
        serving_config=serving_config,
        query=discoveryengine.Query(text=query),
    )


def _extract_reference_chunk_id(reference: Any) -> str | None:
    """Reference shape varies by API surface; checks known candidate field
    paths in order, returns None (not raise) if none match - an unresolvable
    reference is dropped by the caller, same discipline as a Search API miss."""
    struct_data = getattr(reference, "struct_data", None)
    chunk_info = getattr(reference, "chunk_info", None)
    if struct_data is None and chunk_info is not None:
        struct_data = getattr(chunk_info, "struct_data", None)
    if struct_data is not None:
        data = struct_to_dict(struct_data)
        if data.get("chunk_id"):
            return data["chunk_id"]

    document_metadata = getattr(reference, "document_metadata", None) or getattr(
        reference, "unstructured_document_info", None
    )
    if document_metadata is not None:
        doc_ref = getattr(document_metadata, "document", None) or getattr(document_metadata, "uri", None)
        if doc_ref:
            return str(doc_ref).rsplit("/", maxsplit=1)[-1]

    return None


def _build_citations(answer: Any, metadata: pd.DataFrame) -> tuple[list[Citation], list[str]]:
    answer_text = getattr(answer, "answer_text", "") or ""
    references = list(getattr(answer, "references", None) or [])
    raw_citations = list(getattr(answer, "citations", None) or [])

    citations: list[Citation] = []
    hallucinated: list[str] = []
    for raw_citation in raw_citations:
        start = getattr(raw_citation, "start_index", 0) or 0
        end = getattr(raw_citation, "end_index", len(answer_text)) or len(answer_text)
        claim = answer_text[start:end]

        for source in getattr(raw_citation, "sources", None) or []:
            ref_index = getattr(source, "reference_id", None)
            if ref_index is None:
                continue
            try:
                reference = references[int(ref_index)]
            except (TypeError, ValueError, IndexError):
                continue

            chunk_id = _extract_reference_chunk_id(reference)
            if chunk_id is None:
                continue

            chunk = resolve_chunk(chunk_id, metadata)
            if chunk is None:
                hallucinated.append(chunk_id)
                continue

            citations.append(Citation(chunk_id=chunk_id, claim=claim, chunk=chunk))

    return citations, hallucinated


@dataclass
class VertexAnswerer:
    """Mirrors VertexRetriever's load-once/query-cheap shape."""

    config: VertexConfig
    client: VertexAnswerClientLike
    metadata: pd.DataFrame
    request_factory: Callable[[str], Any] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.request_factory is None:
            self.request_factory = lambda query: _build_answer_request(query, self.config)

    @classmethod
    def from_config(cls, config: VertexConfig, index_metadata_path: Path) -> "VertexAnswerer":
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.ConversationalSearchServiceClient()
        metadata = load_metadata(index_metadata_path)
        return cls(config=config, client=client, metadata=metadata)

    def answer(self, query: str) -> tuple[GeneratedAnswer, float | None]:
        request = self.request_factory(query)
        response = self.client.answer_query(request)

        answer = getattr(response, "answer", None)
        if answer is None:
            raise RuntimeError(
                f"Unexpected Vertex AnswerQueryResponse shape - no 'answer' field present: {response!r}"
            )

        answer_text = getattr(answer, "answer_text", "") or ""
        citations, hallucinated = _build_citations(answer, self.metadata)

        generated = GeneratedAnswer(
            query=query,
            answer=answer_text,
            citations=citations,
            # Heuristic pending live-API confirmation (see module docstring): treat a
            # non-empty answer_text as "answered". Vertex may expose a dedicated
            # no-answer signal - verify before trusting this on real out-of-corpus queries.
            answerable=bool(answer_text.strip()),
            hallucinated_citation_ids=hallucinated,
            has_hallucinated_citations=bool(hallucinated),
            model="vertex-answer-api",
            input_tokens=0,
            output_tokens=0,
            stop_reason="vertex_answer",
        )
        grounding_score = getattr(answer, "grounding_score", None)
        return generated, grounding_score
