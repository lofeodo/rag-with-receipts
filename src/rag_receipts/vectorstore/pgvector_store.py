"""pgvector-on-Cloud-SQL VectorStore implementation (pgvector stretch goal).

Connects via the Cloud SQL Python Connector + pg8000, using IAM database
authentication (no password, no Secret Manager secret - see CLAUDE.md's
pgvector stretch goal status note for the justification: this backend is
never deployed to the live Cloud Run service, so the credential only needs
to work for one developer's local scripts during a short provisioning
window, which is exactly what IAM auth - reusing the gcloud ADC login
already set up for this project - is for).

Real-library finding, verified against the installed packages before writing
this module (not assumed from documentation): the Cloud SQL Python Connector's
`driver="pg8000"` path always returns a `pg8000.dbapi.Connection` (the
standard DB-API object, with .cursor()/.execute()/%s placeholders) - never a
`pg8000.native.Connection`. pgvector's own `pgvector.pg8000.register_vector()`
helper requires a `pg8000.native.Connection` (it calls `.run(...)`, which the
DB-API Connection does not have) - so it is NOT compatible with a Connector-
produced connection, despite both living under the pg8000 umbrella. Rather
than fight that mismatch, vectors are passed as pgvector's own text input
format (`'[v0,v1,...]'`, cast to `::vector` directly in the SQL) via ordinary
%s string parameters - no type adapter registration needed at all, for
either direction (search() never selects the embedding column back out, only
the derived score).

Exact search only, by design: no ivfflat/hnsw index is created anywhere in
this module (see SCHEMA_SQL) - the pgvector stretch goal's locked requirement
is that results match FAISS exactly, which only holds for exact search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from rag_receipts.vectorstore.base import ScoredChunk
from rag_receipts.vectorstore.config import PgvectorConfig

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    chunk_id       TEXT PRIMARY KEY,
    page_title     TEXT NOT NULL,
    url            TEXT NOT NULL,
    section_path   TEXT NOT NULL,
    heading_anchor TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    text           TEXT NOT NULL,
    token_count    INTEGER NOT NULL,
    embedding      vector({dim}) NOT NULL
);
"""
# Deliberately no "CREATE INDEX ... USING ivfflat/hnsw ON {table} (embedding ...)"
# anywhere in this module - exact sequential-scan search only (locked decision).


def _vector_literal(vector: np.ndarray) -> str:
    """pgvector's text input format: '[v0,v1,...]'. %.9g keeps full float32
    precision (float32 has ~7 significant decimal digits) without the bulk of
    repr()'s full-precision float64 string."""
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


ConnectFn = Callable[[PgvectorConfig], tuple[Any, Any]]


def _default_connect(config: PgvectorConfig) -> tuple[Any, Any]:
    from google.cloud.sql.connector import Connector

    connector = Connector()
    connection = connector.connect(
        config.instance_connection_name,
        "pg8000",
        user=config.iam_user,
        db=config.database,
        enable_iam_auth=True,
    )
    return connection, connector


@dataclass
class PgvectorStore:
    """Cloud SQL / pgvector backend. One persistent connection per instance,
    no pooling - this backend is never productionized (see module docstring),
    so there is no concurrent-request story to serve; every caller in this
    project (scripts, the eval harness, the equivalence/latency checks) issues
    one query at a time."""

    config: PgvectorConfig
    connection: Any
    connector: Any | None = None

    @classmethod
    def from_config(
        cls, config: PgvectorConfig, *, connect_fn: ConnectFn = _default_connect
    ) -> "PgvectorStore":
        connection, connector = connect_fn(config)
        return cls(config=config, connection=connection, connector=connector)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[ScoredChunk]:
        literal = _vector_literal(query_vector)
        sql = (
            "SELECT chunk_id, page_title, url, section_path, heading_anchor, "
            "source_type, text, token_count, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.config.table_name} "
            "ORDER BY embedding <=> %s::vector ASC "
            "LIMIT %s"
        )
        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, (literal, literal, top_k))
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return [
            ScoredChunk(
                chunk_id=row[0],
                page_title=row[1],
                url=row[2],
                section_path=row[3],
                heading_anchor=row[4],
                source_type=row[5],
                text=row[6],
                token_count=int(row[7]),
                score=float(row[8]),
            )
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()
        if self.connector is not None:
            self.connector.close()


def ensure_schema(connection: Any, config: PgvectorConfig) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(SCHEMA_SQL.format(table=config.table_name, dim=config.embedding_dim))
    finally:
        cursor.close()
    connection.commit()


def upsert_chunks(
    config: PgvectorConfig,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    connect_fn: ConnectFn = _default_connect,
    batch_size: int = 200,
) -> int:
    """Build-path bulk writer, used only by scripts/build_index_pgvector.py.
    Ensures the extension/table exist, then upserts every (metadata row,
    embedding) pair in batches via executemany + ON CONFLICT ... DO UPDATE.
    Returns the number of rows written."""
    connection, connector = connect_fn(config)
    try:
        ensure_schema(connection, config)
        sql = (
            f"INSERT INTO {config.table_name} "
            "(chunk_id, page_title, url, section_path, heading_anchor, source_type, "
            "text, token_count, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector) "
            "ON CONFLICT (chunk_id) DO UPDATE SET "
            "page_title = EXCLUDED.page_title, url = EXCLUDED.url, "
            "section_path = EXCLUDED.section_path, heading_anchor = EXCLUDED.heading_anchor, "
            "source_type = EXCLUDED.source_type, text = EXCLUDED.text, "
            "token_count = EXCLUDED.token_count, embedding = EXCLUDED.embedding"
        )
        cursor = connection.cursor()
        written = 0
        try:
            rows = metadata.reset_index(drop=True)
            for start in range(0, len(rows), batch_size):
                batch_rows = rows.iloc[start : start + batch_size]
                batch_embeddings = embeddings[start : start + batch_size]
                params = [
                    (
                        row.chunk_id,
                        row.page_title,
                        row.url,
                        row.section_path,
                        row.heading_anchor,
                        row.source_type,
                        row.text,
                        int(row.token_count),
                        _vector_literal(vec),
                    )
                    for row, vec in zip(batch_rows.itertuples(), batch_embeddings)
                ]
                cursor.executemany(sql, params)
                connection.commit()
                written += len(params)
        finally:
            cursor.close()
        return written
    finally:
        connection.close()
        if connector is not None:
            connector.close()
