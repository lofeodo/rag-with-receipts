"""Run the retrieval pipeline for a single query.

Usage:
    python scripts/retrieve.py "How do you start Monkey Madness I?"
    python scripts/retrieve.py "..." --config config/config.yaml --top-k 3 --json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from rag_receipts.config import load_config
from rag_receipts.retrieval.pipeline import Retriever

app = typer.Typer(add_completion=False)


@app.command()
def main(
    query: str = typer.Argument(..., help="Query text"),
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
    top_k: int = typer.Option(None, "--top-k", help="Override configured retrieval.top_k_final"),
    as_json: bool = typer.Option(False, "--json", help="Print results as JSON"),
) -> None:
    cfg = load_config(config)
    if top_k is not None:
        cfg.retrieval.top_k_final = top_k

    retriever = Retriever.from_config(cfg)
    results = retriever.retrieve(query)

    if as_json:
        typer.echo(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
        return

    typer.echo(f"Query: {query}\n")
    for r in results:
        typer.echo(
            f"[{r.final_rank}] rerank={r.rerank_score:.3f} dense={r.dense_score:.3f}  "
            f"{r.page_title} > {r.section_path}"
        )
        typer.echo(f"    {r.url}#{r.heading_anchor}")
        snippet = r.text[:200] + ("..." if len(r.text) > 200 else "")
        typer.echo(f"    {snippet}\n")


if __name__ == "__main__":
    app()
