"""FAISS flat-IP index build/save/load helpers (Step 2)."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def build_flat_ip_index(embeddings: np.ndarray) -> faiss.Index:
    """IndexFlatIP over L2-normalized embeddings == cosine similarity search."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings, dtype="float32"))
    return index


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))
