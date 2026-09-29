"""Writers for the indexing pipeline's output artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import pandas as pd

from rag_receipts.indexing.faiss_index import save_index


def write_index(index: faiss.Index, index_dir: Path, filename: str) -> Path:
    path = index_dir / filename
    save_index(index, path)
    return path


def write_metadata(df: pd.DataFrame, index_dir: Path, filename: str) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    path = index_dir / filename
    df.to_parquet(path, index=False)
    return path


def write_stats(stats: dict, stats_path: Path) -> Path:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats_path
