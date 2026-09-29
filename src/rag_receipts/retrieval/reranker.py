"""Cross-encoder reranker wrapper (Step 3).

Scores (query, passage) pairs directly, unlike the bi-encoder Embedder - no vector
index involved, just a relevance score per candidate. The reranker scores the same
breadcrumb-qualified text that was embedded in Step 2 (see rerank_text below), since
OSRS wiki chunks are often terse out of page/section context.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from rag_receipts.indexing.embedder import resolve_device
from rag_receipts.ingestion.utils import breadcrumb
from rag_receipts.retrieval.config import RerankerConfig


class CrossEncoderModel(Protocol):
    """Structural subset of CrossEncoder.predict - lets tests inject a fake."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> np.ndarray: ...


def rerank_text(page_title: str, section_path: str, text: str) -> str:
    """Same text form embedded in Step 2 - breadcrumb + text, not bare text."""
    return f"{breadcrumb(page_title, section_path)}\n\n{text}"


@lru_cache(maxsize=1)
def get_reranker_model(model_name: str, device: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=device)


@dataclass
class Reranker:
    config: RerankerConfig
    model: CrossEncoderModel

    @classmethod
    def from_config(cls, config: RerankerConfig) -> "Reranker":
        device = resolve_device(config.device)
        return cls(config=config, model=get_reranker_model(config.model_name, device))

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
        )
        return np.asarray(scores, dtype="float32")
