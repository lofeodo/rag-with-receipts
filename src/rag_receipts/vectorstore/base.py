"""Backend-agnostic vector search contract (pgvector stretch goal).

ScoredChunk is the single-round-trip search result every VectorStore
implementation returns: chunk_id + full metadata + a cosine-similarity score
(higher = more similar), joined in one step regardless of whether that join
happens in-process against a parquet DataFrame (FaissVectorStore) or inside
one SQL query (PgvectorStore). Distinct from retrieval.models.RetrievedChunk,
which adds rerank_score/dense_rank/final_rank after the reranking stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_receipts.ingestion.models import SourceType


@dataclass
class ScoredChunk:
    chunk_id: str
    page_title: str
    url: str
    section_path: str
    heading_anchor: str
    source_type: SourceType
    text: str
    token_count: int
    score: float


class VectorStore(Protocol):
    """Structural contract satisfied by FaissVectorStore and PgvectorStore.

    query_vector is a single 1D float32 array, shape (dim,) - Retriever only
    ever issues one query at a time, so there is no batch-query method.
    """

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]: ...
