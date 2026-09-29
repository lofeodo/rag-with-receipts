"""Result shape for the retrieval pipeline (Step 3)."""

from __future__ import annotations

from dataclasses import dataclass

from rag_receipts.ingestion.models import SourceType


@dataclass
class RetrievedChunk:
    """A single reranked, citation-ready retrieval result."""

    chunk_id: str
    page_title: str
    url: str
    section_path: str
    heading_anchor: str
    source_type: SourceType
    text: str
    token_count: int
    dense_score: float
    rerank_score: float
    dense_rank: int
    final_rank: int
