"""Parse + chunk cached HTML into parquet/jsonl/stats output.

Usage:
    python scripts/ingest.py --config config/config.yaml
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_receipts.config import load_config
from rag_receipts.ingestion.output import write_chunks, write_sample, write_stats
from rag_receipts.ingestion.pipeline import run_ingest

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    chunks, stats = run_ingest(cfg)

    write_chunks(chunks, Path(cfg.ingestion.processed_dir))
    write_stats(stats, Path(cfg.ingestion.processed_dir))
    write_sample(chunks, Path(cfg.ingestion.results_dir))

    typer.echo(f"Pages parsed: {stats['page_count']}")
    typer.echo(f"Chunks written: {stats['chunk_count']}")
    if stats["parse_failures"]:
        typer.echo(f"Parse failures: {len(stats['parse_failures'])}")
        for failure in stats["parse_failures"]:
            typer.echo(f"  {failure['title']}: {failure['error']}")


if __name__ == "__main__":
    app()
