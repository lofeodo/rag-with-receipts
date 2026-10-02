"""Config for the pgvector-on-Cloud-SQL backend (pgvector stretch goal).

Plain dataclass + PyYAML, matching retrieval/config.py's style (no pydantic).
No dependency on indexing/ - indexing/config.py imports PgvectorConfig from
here and nests it into IndexingConfig, not the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PgvectorConfig:
    enabled_for_build: bool = False
    project_id: str = ""
    region: str = "northamerica-northeast1"
    instance_name: str = "rag-receipts-pgvector"
    database: str = "rag_receipts"
    table_name: str = "chunks"
    iam_user: str = ""
    embedding_dim: int = 1024

    @property
    def instance_connection_name(self) -> str:
        return f"{self.project_id}:{self.region}:{self.instance_name}"


def _build_pgvector_config(raw: dict) -> PgvectorConfig:
    return PgvectorConfig(
        enabled_for_build=raw.get("enabled_for_build", False),
        project_id=raw.get("project_id", ""),
        region=raw.get("region", "northamerica-northeast1"),
        instance_name=raw.get("instance_name", "rag-receipts-pgvector"),
        database=raw.get("database", "rag_receipts"),
        table_name=raw.get("table_name", "chunks"),
        iam_user=raw.get("iam_user", ""),
        embedding_dim=raw.get("embedding_dim", 1024),
    )
