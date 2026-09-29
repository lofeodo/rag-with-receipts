"""Orchestration for the build_index CLI script."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import pandas as pd

from rag_receipts.config import AppConfig
from rag_receipts.indexing.embedder import Embedder, resolve_device
from rag_receipts.indexing.faiss_index import build_flat_ip_index
from rag_receipts.ingestion.utils import breadcrumb

logger = logging.getLogger(__name__)


def load_chunks(chunks_path: Path) -> pd.DataFrame:
    return pd.read_parquet(chunks_path)


def build_embedding_texts(df: pd.DataFrame) -> list[str]:
    """breadcrumb + blank line + chunk text - the exact string that gets encoded."""
    return [
        f"{breadcrumb(row.page_title, row.section_path)}\n\n{row.text}"
        for row in df.itertuples()
    ]


def run_index(
    config: AppConfig, embedder: Embedder | None = None
) -> tuple[faiss.Index, pd.DataFrame, dict]:
    chunks_path = Path(config.ingestion.processed_dir) / "chunks.parquet"
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found - run scripts/ingest.py first")

    df = load_chunks(chunks_path)
    if df.empty:
        raise ValueError(f"{chunks_path} has 0 rows - run scripts/ingest.py first")

    texts = build_embedding_texts(df)
    embedder = embedder or Embedder.from_config(config.indexing.embedding)
    embeddings = embedder.encode_passages(texts)  # row order == df row order, preserved

    index = build_flat_ip_index(embeddings)

    stats = {
        "chunk_count": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_model": config.indexing.embedding.model_name,
        "device": resolve_device(config.indexing.embedding.device),
        "index_type": "flat_ip",
        "batch_size": config.indexing.embedding.batch_size,
        "normalize_embeddings": config.indexing.embedding.normalize_embeddings,
        "source_type_breakdown": df["source_type"].value_counts().to_dict(),
    }
    return index, df, stats
