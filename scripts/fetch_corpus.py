"""Resolve the corpus page list and fetch+cache each page's rendered HTML.

Usage:
    python scripts/fetch_corpus.py --config config/config.yaml [--refresh]
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer

from rag_receipts.ingestion.config import load_config
from rag_receipts.ingestion.pipeline import run_fetch

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-resolve the page list instead of reusing page_list.json"
    ),
) -> None:
    cfg = load_config(config)
    results = run_fetch(cfg, force_refresh=refresh)

    counts = Counter(r.status for r in results)
    typer.echo(
        f"Fetched {len(results)} pages: "
        f"{counts.get('cached', 0)} cached, {counts.get('fetched', 0)} fetched, "
        f"{counts.get('failed', 0)} failed"
    )
    for r in results:
        if r.status == "failed":
            typer.echo(f"  FAILED: {r.title} - {r.error}")


if __name__ == "__main__":
    app()
