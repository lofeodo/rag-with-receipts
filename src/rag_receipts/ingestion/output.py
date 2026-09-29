"""Writers for the ingestion pipeline's output artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rag_receipts.ingestion.models import Chunk


def write_chunks(chunks: list[Chunk], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(c) for c in chunks])
    df.to_parquet(processed_dir / "chunks.parquet", index=False)
    df.to_json(processed_dir / "chunks.jsonl", orient="records", lines=True, force_ascii=False)


def write_stats(stats: dict, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "ingest_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_sample(chunks: list[Chunk], results_dir: Path, n: int = 30) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    sample = chunks[:n]
    with (results_dir / "sample_chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in sample:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
