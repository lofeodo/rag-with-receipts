"""Run a handful of hand-picked queries through the retrieval pipeline and write
results/sample_retrievals.json - a committed, small visibility artifact showing
retrieval quality without requiring the reader to run the pipeline themselves
(mirrors Step 1's sample_chunks.jsonl / Step 2's index_stats.json).

Usage:
    python scripts/sample_retrievals.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_receipts.config import load_config
from rag_receipts.retrieval.pipeline import Retriever

QUERIES = [
    "What items are required to start Monkey Madness I?",
    "How much Slayer experience is needed for level 70?",
    "What is the Abyssal whip's special attack?",
    "What potions require Herblore level 78?",
    "How do you defeat Zooknock in Monkey Madness I?",
]

OUTPUT_PATH = Path("results/sample_retrievals.json")


def main() -> None:
    cfg = load_config("config/config.yaml")
    retriever = Retriever.from_config(cfg)

    samples = [
        {"query": query, "results": [asdict(r) for r in retriever.retrieve(query)]}
        for query in QUERIES
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} sample retrievals to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
