"""Config for the embedding + FAISS indexing pipeline.

Plain dataclasses + PyYAML, matching ingestion/config.py's style (no pydantic).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-large-en-v1.5"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "auto"  # auto | cpu | cuda
    show_progress_bar: bool = True
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class IndexOutputConfig:
    index_dir: str = "data/index"
    index_filename: str = "faiss.index"
    metadata_filename: str = "index_metadata.parquet"
    stats_path: str = "data/processed/index_stats.json"


@dataclass
class IndexingConfig:
    vector_index: str = "faiss"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    output: IndexOutputConfig = field(default_factory=IndexOutputConfig)


def _build_indexing_config(raw: dict) -> IndexingConfig:
    output = raw.get("output", {})
    embedding = EmbeddingConfig(
        model_name=raw.get("embedding_model", "BAAI/bge-large-en-v1.5"),
        batch_size=raw.get("batch_size", 32),
        normalize_embeddings=raw.get("normalize_embeddings", True),
        device=raw.get("device", "auto"),
        show_progress_bar=raw.get("show_progress_bar", True),
        query_instruction=raw.get(
            "query_instruction",
            "Represent this sentence for searching relevant passages: ",
        ),
    )
    output_config = IndexOutputConfig(
        index_dir=output.get("index_dir", "data/index"),
        index_filename=output.get("index_filename", "faiss.index"),
        metadata_filename=output.get("metadata_filename", "index_metadata.parquet"),
        stats_path=output.get("stats_path", "data/processed/index_stats.json"),
    )
    return IndexingConfig(
        vector_index=raw.get("vector_index", "faiss"),
        embedding=embedding,
        output=output_config,
    )
