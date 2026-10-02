"""Real Cloud SQL / pgvector integration test (pgvector stretch goal).

Only runnable during the live provision-demo window - gated on
PGVECTOR_INSTANCE_CONNECTION_NAME (mirroring test_generator_integration.py's
ANTHROPIC_API_KEY gate), since (unlike the other `slow` tests, which just
load a real local ML model) this one needs a real, billed, provisioned Cloud
SQL instance to exist.

Uses its own throwaway table (not the real `chunks` table the equivalence
script/corpus build populates), dropped at the end - this test never touches
the real corpus data.

Run explicitly (after `gcloud auth application-default login` and with the
instance up):
    PGVECTOR_INSTANCE_CONNECTION_NAME=<project>:<region>:<instance> \
        pytest -q -m slow tests/vectorstore/test_pgvector_integration.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from rag_receipts.vectorstore.config import PgvectorConfig
from rag_receipts.vectorstore.pgvector_store import PgvectorStore, ensure_schema, upsert_chunks

pytestmark = pytest.mark.slow

_INSTANCE_CONNECTION_NAME = os.environ.get("PGVECTOR_INSTANCE_CONNECTION_NAME", "")
_IAM_USER = os.environ.get("PGVECTOR_IAM_USER", "")


def _real_config() -> PgvectorConfig:
    project_id, region, instance_name = _INSTANCE_CONNECTION_NAME.split(":")
    return PgvectorConfig(
        project_id=project_id,
        region=region,
        instance_name=instance_name,
        database="rag_receipts",
        table_name="chunks_test_pgvector_integration",
        iam_user=_IAM_USER,
        embedding_dim=4,
    )


@pytest.mark.skipif(
    not _INSTANCE_CONNECTION_NAME,
    reason="PGVECTOR_INSTANCE_CONNECTION_NAME not set - skipping real Cloud SQL call",
)
class TestRealPgvectorInstance:
    def test_iam_connection_extension_self_similarity_and_exact_search(self):
        config = _real_config()
        store = PgvectorStore.from_config(config)
        try:
            ensure_schema(store.connection, config)

            metadata = pd.DataFrame(
                [
                    {
                        "chunk_id": "pgvector_itest__0",
                        "page_title": "Test",
                        "url": "https://example.invalid/0",
                        "section_path": "",
                        "heading_anchor": "",
                        "source_type": "prose",
                        "text": "chunk zero",
                        "token_count": 2,
                    },
                    {
                        "chunk_id": "pgvector_itest__1",
                        "page_title": "Test",
                        "url": "https://example.invalid/1",
                        "section_path": "",
                        "heading_anchor": "",
                        "source_type": "prose",
                        "text": "chunk one",
                        "token_count": 2,
                    },
                ]
            )
            embeddings = np.eye(2, 4, dtype="float32")  # 2 rows, 4-dim, orthonormal
            written = upsert_chunks(config, metadata, embeddings)
            assert written == 2

            # Self-similarity: querying with a known stored vector returns it as
            # top-1 with score ~= 1.0 - the pgvector analogue of Step 2's FAISS
            # self-similarity sanity check.
            results = store.search(embeddings[0], top_k=2)
            assert results[0].chunk_id == "pgvector_itest__0"
            assert abs(results[0].score - 1.0) < 1e-5

            # EXPLAIN confirms a sequential scan, not an index scan - automated
            # proof that no ivfflat/hnsw index exists (exact search only, per
            # the locked decision), not just the absence of a CREATE INDEX line
            # in source.
            cursor = store.connection.cursor()
            try:
                literal = "[1,0,0,0]"
                cursor.execute(
                    f"EXPLAIN SELECT chunk_id FROM {config.table_name} "
                    f"ORDER BY embedding <=> '{literal}'::vector ASC LIMIT 2"
                )
                plan = "\n".join(row[0] for row in cursor.fetchall())
            finally:
                cursor.close()
            assert "Seq Scan" in plan
            assert "Index Scan" not in plan
        finally:
            cursor = store.connection.cursor()
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {config.table_name}")
                store.connection.commit()
            finally:
                cursor.close()
            store.close()
