"""FAISS-backed VectorStore implementation.

dense_search/load_metadata are the same functions previously defined inline in
retrieval/pipeline.py (Step 3) - moved here so retrieval/pipeline.py can be
backend-agnostic, calling only VectorStore.search() regardless of whether the
backend is FAISS or pgvector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from rag_receipts.indexing.config import IndexingConfig
from rag_receipts.indexing.faiss_index import load_index
from rag_receipts.vectorstore.base import ScoredChunk


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


@dataclass
class FaissVectorStore:
    """In-process flat-index backend - the project's default. Row i of `index`
    is aligned to row i of `metadata` (FAISS flat indices only support int64
    row ids, not string chunk_ids), so .search() joins the two by row position."""

    index: faiss.Index
    metadata: pd.DataFrame

    @classmethod
    def from_config(cls, config: IndexingConfig) -> "FaissVectorStore":
        index_dir = Path(config.output.index_dir)
        index_path = index_dir / config.output.index_filename
        if not index_path.exists():
            raise FileNotFoundError(f"{index_path} not found - run scripts/build_index.py first")
        return cls(
            index=load_index(index_path),
            metadata=load_metadata(index_dir / config.output.metadata_filename),
        )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        scores, ids = dense_search(self.index, query_vector.reshape(1, -1), top_k)
        valid = [(score, idx) for score, idx in zip(scores[0], ids[0]) if idx != -1]
        if not valid:
            return []
        rows = self.metadata.iloc[[int(idx) for _, idx in valid]].reset_index(drop=True)
        return [
            ScoredChunk(
                chunk_id=row.chunk_id,
                page_title=row.page_title,
                url=row.url,
                section_path=row.section_path,
                heading_anchor=row.heading_anchor,
                source_type=row.source_type,
                text=row.text,
                token_count=int(row.token_count),
                score=float(score),
            )
            for (score, _idx), row in zip(valid, rows.itertuples())
        ]
