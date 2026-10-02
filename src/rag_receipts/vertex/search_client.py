"""Config A: Vertex AI Search's Search API as a retrieval-only drop-in for the
hand-built Retriever - isolates retrieval quality against the existing
pipeline on equal footing (see CLAUDE.md's Vertex stretch-goal status note).

VertexRetriever.retrieve(query) has the exact same signature as
Retriever.retrieve(query) (str in, list[RetrievedChunk] out), so it slots
directly into Generator.generate(query, chunks) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from rag_receipts.retrieval.models import RetrievedChunk
from rag_receipts.vectorstore.faiss_store import load_metadata
from rag_receipts.vertex.config import VertexConfig
from rag_receipts.vertex.resolve import resolve_chunk, struct_to_dict


class VertexSearchClientLike(Protocol):
    def search(self, request: Any) -> Iterable[Any]: ...


def _build_search_request(query: str, config: VertexConfig) -> Any:
    from google.cloud import discoveryengine_v1 as discoveryengine

    serving_config = (
        f"projects/{config.project_id}/locations/{config.location}/"
        f"collections/default_collection/dataStores/{config.data_store_id}/"
        f"servingConfigs/{config.search_serving_config}"
    )
    return discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=query,
        page_size=config.search_top_k,
    )


def _resolve_chunk_id(result: Any) -> str | None:
    """Cross-checks document.id against structData.chunk_id - two independent
    channels of the same value (see upload.py's manifest-building docstring),
    so a transformation/escaping quirk in either channel alone doesn't silently
    drop a resolvable chunk."""
    document = result.document
    struct_data = struct_to_dict(document.struct_data)
    struct_chunk_id = struct_data.get("chunk_id")
    document_id = document.id

    if struct_chunk_id and struct_chunk_id == document_id:
        return struct_chunk_id
    # Fall back to whichever channel is present if they disagree or one is missing -
    # prefer structData since it's the field this project controls end-to-end.
    return struct_chunk_id or document_id or None


def parse_search_results(
    results: Iterable[Any], metadata: pd.DataFrame, *, top_k: int
) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for dense_rank, result in enumerate(results, start=1):
        chunk_id = _resolve_chunk_id(result)
        if chunk_id is None:
            continue
        chunk = resolve_chunk(chunk_id, metadata, dense_rank=dense_rank, final_rank=dense_rank)
        if chunk is None:
            continue
        chunks.append(chunk)
        if len(chunks) >= top_k:
            break
    return chunks


@dataclass
class VertexRetriever:
    """Mirrors Retriever's load-once/query-cheap shape: construct once via
    from_config, .retrieve() does no I/O beyond the search call itself."""

    config: VertexConfig
    client: VertexSearchClientLike
    metadata: pd.DataFrame
    request_factory: Callable[[str], Any] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.request_factory is None:
            self.request_factory = lambda query: _build_search_request(query, self.config)

    @classmethod
    def from_config(cls, config: VertexConfig, index_metadata_path: Path) -> "VertexRetriever":
        from google.cloud import discoveryengine_v1 as discoveryengine

        client = discoveryengine.SearchServiceClient()
        metadata = load_metadata(index_metadata_path)
        return cls(config=config, client=client, metadata=metadata)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        request = self.request_factory(query)
        results = self.client.search(request)
        return parse_search_results(results, self.metadata, top_k=self.config.search_top_k)
