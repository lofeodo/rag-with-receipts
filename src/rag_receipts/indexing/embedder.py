"""BGE passage/query embedding wrapper (Step 2).

BGE models are trained for cosine similarity: encode with normalize_embeddings=True
and pair with a FAISS IndexFlatIP (inner product on unit vectors == cosine similarity).
Passage text gets no prefix; query text gets a retrieval instruction prefix - this
asymmetry is BGE-specific and is the reason encode_passages/encode_queries are two
methods instead of one, even though only encode_passages is used until Step 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag_receipts.indexing.config import EmbeddingConfig


class Encoder(Protocol):
    """Structural subset of SentenceTransformer.encode - lets tests inject a fake."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


@dataclass
class Embedder:
    config: EmbeddingConfig
    model: Encoder

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "Embedder":
        device = resolve_device(config.device)
        return cls(config=config, model=get_model(config.model_name, device))

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=self.config.show_progress_bar,
            ),
            dtype="float32",
        )

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        prefixed = [self.config.query_instruction + q for q in queries]
        return np.asarray(
            self.model.encode(
                prefixed,
                batch_size=self.config.batch_size,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=self.config.show_progress_bar,
            ),
            dtype="float32",
        )
