"""Result shape for timed retrieval (Step 7)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalTiming:
    embed_query_s: float
    dense_search_s: float
    rerank_s: float
    total_s: float
