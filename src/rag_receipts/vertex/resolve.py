"""Resolves a Vertex-returned chunk_id back to this project's own chunk text/metadata.

A chunk_id coming back from Vertex AI Search is untrusted input, exactly like an
LLM-cited chunk_id in Step 4's generation.py - it names a chunk our own index
once produced, but isn't guaranteed to still resolve (a typo, a stale id, a
non-chunk document somehow in the response). resolve_chunk() returns None
rather than raising so callers can drop and count unresolvable ids instead of
crashing a whole eval run on one bad citation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from rag_receipts.retrieval.models import RetrievedChunk


def struct_to_dict(struct_data: Any) -> dict:
    """struct_data is a protobuf Struct on a real Discovery Engine response;
    unit tests hand back a plain dict directly, so this short-circuits rather
    than requiring the real protobuf type to be installed."""
    if isinstance(struct_data, dict):
        return struct_data
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(struct_data)


def resolve_chunk(
    chunk_id: str,
    metadata: pd.DataFrame,
    *,
    dense_score: float = 0.0,
    rerank_score: float = 0.0,
    dense_rank: int = 0,
    final_rank: int = 0,
) -> RetrievedChunk | None:
    """Looks up chunk_id in index_metadata.parquet and builds a real RetrievedChunk.

    dense_score/rerank_score default to 0.0 placeholders - Vertex has no
    equivalent of our local dense/rerank scores, and score_retrieval() never
    reads these fields, so the placeholder can't distort any retrieval metric.
    dense_rank/final_rank should be set by the caller from Vertex's own
    response order.
    """
    matches = metadata.loc[metadata["chunk_id"] == chunk_id]
    if matches.empty:
        return None

    row = matches.iloc[0]
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        page_title=row.page_title,
        url=row.url,
        section_path=row.section_path,
        heading_anchor=row.heading_anchor,
        source_type=row.source_type,
        text=row.text,
        token_count=int(row.token_count),
        dense_score=dense_score,
        rerank_score=rerank_score,
        dense_rank=dense_rank,
        final_rank=final_rank,
    )
