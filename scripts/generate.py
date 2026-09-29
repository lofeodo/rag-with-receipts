"""Run retrieval + generation for a single query, printing a grounded, cited answer.

Usage:
    python scripts/generate.py "How do you start Monkey Madness I?"
    python scripts/generate.py "..." --config config/config.yaml --json
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

from rag_receipts.config import load_config
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.pipeline import Retriever

app = typer.Typer(add_completion=False)


@app.command()
def main(
    query: str = typer.Argument(..., help="Query text"),
    config: Path = typer.Option(Path("config/config.yaml"), "--config", help="Path to config.yaml"),
    as_json: bool = typer.Option(False, "--json", help="Print result as JSON"),
) -> None:
    cfg = load_config(config)

    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)

    chunks = retriever.retrieve(query)
    answer = generator.generate(query, chunks)

    if as_json:
        typer.echo(json.dumps(asdict(answer), indent=2, ensure_ascii=False))
        return

    typer.echo(f"Query: {query}\n")

    if not answer.answerable:
        typer.echo("NOT ANSWERABLE from the retrieved context.")
    typer.echo(f"{answer.answer}\n")

    if answer.citations:
        typer.echo("Citations:")
        for i, citation in enumerate(answer.citations, start=1):
            chunk = citation.chunk
            typer.echo(f"  [{i}] {chunk.page_title} > {chunk.section_path}")
            typer.echo(f"      {chunk.url}#{chunk.heading_anchor}")
            typer.echo(f"      supports: {citation.claim}")

    if answer.has_hallucinated_citations:
        typer.echo(
            f"\nWARNING: {len(answer.hallucinated_citation_ids)} citation(s) referenced a "
            f"chunk_id not in the retrieved set and were dropped: "
            f"{answer.hallucinated_citation_ids}"
        )


if __name__ == "__main__":
    app()
