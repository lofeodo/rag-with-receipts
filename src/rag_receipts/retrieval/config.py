"""Config for the retrieval pipeline (dense search + cross-encoder reranking).

Plain dataclasses + PyYAML, matching indexing/config.py's style (no pydantic).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-base"
    batch_size: int = 32
    device: str = "auto"  # auto | cpu | cuda
    show_progress_bar: bool = False


@dataclass
class RetrievalConfig:
    top_k_dense: int = 30
    top_k_final: int = 5
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


def _build_retrieval_config(raw: dict) -> RetrievalConfig:
    reranker = RerankerConfig(
        model_name=raw.get("reranker_model", "BAAI/bge-reranker-base"),
        batch_size=raw.get("reranker_batch_size", 32),
        device=raw.get("reranker_device", "auto"),
        show_progress_bar=raw.get("reranker_show_progress_bar", False),
    )
    return RetrievalConfig(
        top_k_dense=raw.get("top_k_dense", 30),
        top_k_final=raw.get("top_k_final", 5),
        reranker=reranker,
    )
