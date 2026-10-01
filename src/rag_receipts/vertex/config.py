"""Config for the Vertex AI Search comparison stretch goal.

Plain dataclass + PyYAML, matching retrieval/config.py's style (no pydantic).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VertexConfig:
    project_id: str = ""
    location: str = "global"
    data_store_id: str = "rag-receipts-corpus"
    engine_id: str = "rag-receipts-search-app"
    search_serving_config: str = "default_search"
    answer_serving_config: str = "default_serving_config"
    search_top_k: int = 10
    gcs_bucket: str = ""
    gcs_prefix: str = "corpus"
    search_results_path: str = "results/vertex_search_eval.json"
    answer_results_path: str = "results/vertex_answer_eval.json"


def _build_vertex_config(raw: dict) -> VertexConfig:
    return VertexConfig(
        project_id=raw.get("project_id", ""),
        location=raw.get("location", "global"),
        data_store_id=raw.get("data_store_id", "rag-receipts-corpus"),
        engine_id=raw.get("engine_id", "rag-receipts-search-app"),
        search_serving_config=raw.get("search_serving_config", "default_search"),
        answer_serving_config=raw.get("answer_serving_config", "default_serving_config"),
        search_top_k=raw.get("search_top_k", 10),
        gcs_bucket=raw.get("gcs_bucket", ""),
        gcs_prefix=raw.get("gcs_prefix", "corpus"),
        search_results_path=raw.get("search_results_path", "results/vertex_search_eval.json"),
        answer_results_path=raw.get("answer_results_path", "results/vertex_answer_eval.json"),
    )
