"""Embed chunks and upsert them into the pgvector-on-Cloud-SQL table
(pgvector stretch goal's build path - a separate script from build_index.py,
not a branch inside it, matching the Vertex stretch's own precedent:
backend-specific persistence that shares nothing operationally with the
FAISS build path, which stays completely untouched by this script).

Reuses indexing/pipeline.py's load_chunks/build_embedding_texts (already
backend-agnostic) and Embedder - only the persistence tail differs from
scripts/build_index.py (upsert_chunks instead of build_flat_ip_index +
write_index/write_metadata).

Refuses to run unless indexing.pgvector.enabled_for_build is explicitly set
to true in config.yaml - a stale `vector_index: pgvector` left in a config
after the Cloud SQL instance has been torn down should fail with a clear
message, not a confusing connection error.

Usage:
    python scripts/build_index_pgvector.py --config config/config.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_receipts.config import load_config
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.indexing.pipeline import build_embedding_texts, load_chunks
from rag_receipts.vectorstore.pgvector_store import upsert_chunks

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    pgvector_cfg = cfg.indexing.pgvector

    if not pgvector_cfg.enabled_for_build:
        raise SystemExit(
            "indexing.pgvector.enabled_for_build is False - refusing to write to a "
            "live Cloud SQL instance without an explicit opt-in in config.yaml."
        )

    chunks_path = Path(cfg.ingestion.processed_dir) / "chunks.parquet"
    if not chunks_path.exists():
        raise SystemExit(f"{chunks_path} not found - run scripts/ingest.py first")

    df = load_chunks(chunks_path)
    typer.echo(f"Loaded {len(df)} chunks from {chunks_path}")

    texts = build_embedding_texts(df)
    embedder = Embedder.from_config(cfg.indexing.embedding)
    embeddings = embedder.encode_passages(texts)
    typer.echo(f"Embedded {len(texts)} chunks (dim={embeddings.shape[1]})")

    written = upsert_chunks(pgvector_cfg, df, embeddings)

    typer.echo(
        f"Upserted {written} rows into "
        f"{pgvector_cfg.instance_connection_name}/{pgvector_cfg.database}.{pgvector_cfg.table_name}"
    )


if __name__ == "__main__":
    app()
