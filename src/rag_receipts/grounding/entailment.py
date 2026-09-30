"""NLI cross-encoder entailment checker (Step 6).

Mirrors retrieval/reranker.py::Reranker's shape (Protocol + lru_cache'd loader +
from_config), scoring (premise, hypothesis) pairs for entailment/contradiction/neutral
instead of (query, passage) relevance.

Label-order safety: NLI cross-encoder checkpoints don't share one standardized
contradiction/entailment/neutral index order (e.g. cross-encoder/nli-deberta-v3-base
uses {0: contradiction, 1: entailment, 2: neutral} - confirmed via its HF config at
implementation time, not assumed). Hardcoding an index order would silently mis-score
if config.grounding.nli_model is ever pointed at a checkpoint with a different order, so
EntailmentChecker resolves a name -> column index map once at load time (from the real
model's id2label) and always returns scores in a fixed (entailment, contradiction,
neutral) column order from score(), regardless of the underlying checkpoint's native
order.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from rag_receipts.grounding.config import GroundingConfig
from rag_receipts.indexing.embedder import resolve_device


class CrossEncoderModel(Protocol):
    """Structural subset of CrossEncoder.predict - lets tests inject a fake."""

    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        apply_softmax: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray: ...


@lru_cache(maxsize=1)
def get_entailment_model(model_name: str, device: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=device)


def resolve_label_index(id2label: dict[int, str]) -> dict[str, int]:
    """Builds a {label_name.lower(): column_index} map from a model's id2label, so
    score() can always return columns in a fixed order regardless of the checkpoint's
    native index order."""
    return {name.lower(): idx for idx, name in id2label.items()}


@dataclass
class EntailmentChecker:
    config: GroundingConfig
    model: CrossEncoderModel
    label_index: dict[str, int]  # resolved label_name -> native column index

    @classmethod
    def from_config(cls, config: GroundingConfig) -> "EntailmentChecker":
        device = resolve_device(config.nli_device)
        model = get_entailment_model(config.nli_model, device)
        label_index = resolve_label_index(model.model.config.id2label)
        return cls(config=config, model=model, label_index=label_index)

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Returns an (n, 3) array with columns fixed as
        (entailment_prob, contradiction_prob, neutral_prob), regardless of the
        underlying model's native column order."""
        raw = self.model.predict(
            pairs,
            batch_size=self.config.nli_batch_size,
            show_progress_bar=self.config.nli_show_progress_bar,
            apply_softmax=True,
            convert_to_numpy=True,
        )
        raw = np.asarray(raw, dtype="float32")
        order = [self.label_index["entailment"], self.label_index["contradiction"], self.label_index["neutral"]]
        return raw[:, order]
