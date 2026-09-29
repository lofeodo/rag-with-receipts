"""Embed chunks and build the FAISS flat index.

Usage:
    python scripts/build_index.py --config config/config.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_receipts.config import load_config
from rag_receipts.indexing.output import write_index, write_metadata, write_stats
from rag_receipts.indexing.pipeline import run_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    index, metadata, stats = run_index(cfg)

    index_dir = Path(cfg.indexing.output.index_dir)
    index_path = write_index(index, index_dir, cfg.indexing.output.index_filename)
    metadata_path = write_metadata(metadata, index_dir, cfg.indexing.output.metadata_filename)
    write_stats(stats, Path(cfg.indexing.output.stats_path))

    typer.echo(
        f"Indexed {stats['chunk_count']} chunks "
        f"(dim={stats['embedding_dim']}, device={stats['device']})"
    )
    typer.echo(f"Index: {index_path}")
    typer.echo(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    app()
