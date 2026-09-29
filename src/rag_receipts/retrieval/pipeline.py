"""Dense search + cross-encoder rerank orchestration (Step 3).

Retriever separates expensive load-once state (FAISS index, metadata, embedding
model, reranker model) from cheap per-query retrieval, so Step 4/5/8 can construct
one Retriever and call .retrieve() in a loop instead of reloading models per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from rag_receipts.config import AppConfig
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.indexing.faiss_index import load_index
from rag_receipts.retrieval.config import RetrievalConfig
from rag_receipts.retrieval.models import RetrievedChunk
from rag_receipts.retrieval.reranker import Reranker, rerank_text


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found - run scripts/build_index.py first")
    return pd.read_parquet(metadata_path)


def dense_search(
    index: faiss.Index, query_vectors: np.ndarray, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Thin pass-through to index.search. (scores, ids), each shape (n_queries, top_k).

    ids contains -1 padding in trailing columns when top_k > index.ntotal - callers
    must filter these before indexing into metadata.
    """
    return index.search(np.ascontiguousarray(query_vectors, dtype="float32"), top_k)


def run_retrieval(
    query: str,
    *,
    index: faiss.Index,
    metadata: pd.DataFrame,
    embedder: Embedder,
    reranker: Reranker,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    query_vecs = embedder.encode_queries([query])
    scores, ids = dense_search(index, query_vecs, config.top_k_dense)
    dense_scores, dense_ids = scores[0], ids[0]

    valid = [(score, idx) for score, idx in zip(dense_scores, dense_ids) if idx != -1]
    if not valid:
        return []

    rows = metadata.iloc[[int(idx) for _, idx in valid]].reset_index(drop=True)
    passages = [rerank_text(row.page_title, row.section_path, row.text) for row in rows.itertuples()]
    rerank_scores = reranker.score(query, passages)

    candidates = [
        RetrievedChunk(
            chunk_id=row.chunk_id,
            page_title=row.page_title,
            url=row.url,
            section_path=row.section_path,
            heading_anchor=row.heading_anchor,
            source_type=row.source_type,
            text=row.text,
            token_count=int(row.token_count),
            dense_score=float(dense_score),
            rerank_score=float(rerank_score),
            dense_rank=dense_rank,
            final_rank=0,
        )
        for dense_rank, ((dense_score, _idx), row, rerank_score) in enumerate(
            zip(valid, rows.itertuples(), rerank_scores), start=1
        )
    ]

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    top = candidates[: config.top_k_final]
    for rank, chunk in enumerate(top, start=1):
        chunk.final_rank = rank
    return top


@dataclass
class Retriever:
    """Owns all expensive load-once state. Construct once via from_config;
    .retrieve() does zero I/O or model loading per call."""

    config: RetrievalConfig
    index: faiss.Index
    metadata: pd.DataFrame
    embedder: Embedder
    reranker: Reranker

    @classmethod
    def from_config(cls, config: AppConfig) -> "Retriever":
        index_dir = Path(config.indexing.output.index_dir)
        index_path = index_dir / config.indexing.output.index_filename
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found - run scripts/build_index.py first")
        index = load_index(index_path)
        metadata = load_metadata(index_dir / config.indexing.output.metadata_filename)
        embedder = Embedder.from_config(config.indexing.embedding)
        reranker = Reranker.from_config(config.retrieval.reranker)
        return cls(
            config=config.retrieval,
            index=index,
            metadata=metadata,
            embedder=embedder,
            reranker=reranker,
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return run_retrieval(
            query,
            index=self.index,
            metadata=self.metadata,
            embedder=self.embedder,
            reranker=self.reranker,
            config=self.config,
        )
