from __future__ import annotations

import numpy as np
import pandas as pd

from rag_receipts.vectorstore.config import PgvectorConfig
from rag_receipts.vectorstore.pgvector_store import PgvectorStore, _vector_literal, upsert_chunks


class FakeCursor:
    def __init__(self, connection: "FakeConnection"):
        self.connection = connection
        self.executed: list[tuple[str, tuple]] = []
        self.executemany_calls: list[tuple[str, list[tuple]]] = []
        self.closed = False
        self._fetch_rows: list[tuple] = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.executed.append((sql, params))
        self._fetch_rows = self.connection.rows_to_return

    def executemany(self, sql: str, param_sets: list[tuple]) -> None:
        self.executemany_calls.append((sql, list(param_sets)))

    def fetchall(self) -> list[tuple]:
        return self._fetch_rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, rows_to_return: list[tuple] | None = None):
        self.rows_to_return = rows_to_return or []
        self.cursors: list[FakeCursor] = []
        self.commits = 0
        self.closed = False

    def cursor(self) -> FakeCursor:
        cursor = FakeCursor(self)
        self.cursors.append(cursor)
        return cursor

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _config(**overrides) -> PgvectorConfig:
    defaults = dict(
        project_id="proj",
        region="northamerica-northeast1",
        instance_name="inst",
        database="db",
        table_name="chunks",
        iam_user="user@example.com",
        embedding_dim=4,
    )
    defaults.update(overrides)
    return PgvectorConfig(**defaults)


def _row(chunk_id: str, score: float) -> tuple:
    return (chunk_id, "Page", "https://example.com", "Section", "anchor", "prose", "text", 10, score)


def test_vector_literal_format():
    literal = _vector_literal(np.array([1.0, 0.5, -0.25], dtype="float32"))

    assert literal == "[1,0.5,-0.25]"


def test_from_config_passes_iam_auth_and_identity_through():
    captured = {}

    def connect_fn(config):
        captured["config"] = config
        return FakeConnection(), FakeConnector()

    store = PgvectorStore.from_config(_config(), connect_fn=connect_fn)

    assert captured["config"].iam_user == "user@example.com"
    assert captured["config"].database == "db"
    assert isinstance(store.connection, FakeConnection)
    assert isinstance(store.connector, FakeConnector)


def test_search_builds_sql_with_vector_casts_and_placeholders():
    connection = FakeConnection(rows_to_return=[_row("c1", 0.9), _row("c2", 0.5)])
    store = PgvectorStore(config=_config(), connection=connection)

    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=2)

    sql, params = connection.cursors[0].executed[0]
    assert sql.count("<=>") == 2
    assert "::vector" in sql
    assert "%s" in sql
    assert params[2] == 2  # top_k passed through, not interpolated into the SQL text
    assert connection.cursors[0].closed is True
    assert [r.chunk_id for r in results] == ["c1", "c2"]
    assert [r.score for r in results] == [0.9, 0.5]


def test_search_returns_scored_chunk_fields_from_row():
    connection = FakeConnection(rows_to_return=[_row("c1", 0.75)])
    store = PgvectorStore(config=_config(), connection=connection)

    results = store.search(np.zeros(4, dtype="float32"), top_k=1)

    chunk = results[0]
    assert chunk.chunk_id == "c1"
    assert chunk.page_title == "Page"
    assert chunk.url == "https://example.com"
    assert chunk.section_path == "Section"
    assert chunk.heading_anchor == "anchor"
    assert chunk.source_type == "prose"
    assert chunk.text == "text"
    assert chunk.token_count == 10
    assert chunk.score == 0.75


def test_close_closes_both_connection_and_connector():
    connection = FakeConnection()
    connector = FakeConnector()
    store = PgvectorStore(config=_config(), connection=connection, connector=connector)

    store.close()

    assert connection.closed is True
    assert connector.closed is True


def test_close_without_connector_does_not_raise():
    connection = FakeConnection()
    store = PgvectorStore(config=_config(), connection=connection, connector=None)

    store.close()

    assert connection.closed is True


def test_upsert_chunks_batches_and_commits_per_batch():
    connection = FakeConnection()

    def connect_fn(config):
        return connection, FakeConnector()

    metadata = pd.DataFrame(
        [
            {
                "chunk_id": f"c{i}",
                "page_title": "Page",
                "url": "https://example.com",
                "section_path": "",
                "heading_anchor": "",
                "source_type": "prose",
                "text": f"text {i}",
                "token_count": 5,
            }
            for i in range(5)
        ]
    )
    embeddings = np.eye(5, dtype="float32")

    written = upsert_chunks(
        _config(), metadata, embeddings, connect_fn=connect_fn, batch_size=2
    )

    assert written == 5
    cursor = connection.cursors[-1]
    # ensure_schema's cursor + the upsert cursor: 3 batches of executemany (2,2,1)
    assert len(cursor.executemany_calls) == 3
    assert [len(params) for _, params in cursor.executemany_calls] == [2, 2, 1]
    assert connection.commits >= 3  # schema commit + one per batch
    assert connection.closed is True


def test_upsert_chunks_sql_casts_embedding_and_upserts_on_conflict():
    connection = FakeConnection()

    def connect_fn(config):
        return connection, FakeConnector()

    metadata = pd.DataFrame(
        [
            {
                "chunk_id": "c0",
                "page_title": "Page",
                "url": "https://example.com",
                "section_path": "",
                "heading_anchor": "",
                "source_type": "prose",
                "text": "text",
                "token_count": 5,
            }
        ]
    )
    embeddings = np.array([[1.0, 0.0]], dtype="float32")

    upsert_chunks(_config(embedding_dim=2), metadata, embeddings, connect_fn=connect_fn)

    cursor = connection.cursors[-1]
    sql, params = cursor.executemany_calls[0]
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql
    assert "::vector" in sql
    assert params[0][-1] == "[1,0]"
