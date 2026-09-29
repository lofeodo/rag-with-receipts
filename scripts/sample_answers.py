"""Run a handful of hand-picked queries end-to-end (retrieval + generation) and write
results/sample_answers.json - a committed, small visibility artifact showing grounded,
cited answers without requiring the reader to run the pipeline themselves (mirrors
Step 3's sample_retrievals.json).

Duplicates Step 3's QUERIES list (rather than importing scripts.sample_retrievals -
scripts/ has no __init__.py, so a `python scripts/sample_answers.py` invocation puts
scripts/ itself, not the repo root, on sys.path, and a cross-script import doesn't
reliably resolve) and adds one deliberately out-of-corpus query to demonstrate the
answerable=false path in the committed sample output.

Usage:
    python scripts/sample_answers.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_receipts.config import load_config
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.pipeline import Retriever

QUERIES = [
    "What items are required to start Monkey Madness I?",
    "How much Slayer experience is needed for level 70?",
    "What is the Abyssal whip's special attack?",
    "What potions require Herblore level 78?",
    "How do you defeat Zooknock in Monkey Madness I?",
    "What are the requirements to start Dragon Slayer II?",  # out of corpus scope
]

OUTPUT_PATH = Path("results/sample_answers.json")


def main() -> None:
    cfg = load_config("config/config.yaml")
    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)

    samples = []
    for query in QUERIES:
        chunks = retriever.retrieve(query)
        answer = generator.generate(query, chunks)
        samples.append({"query": query, "answer": asdict(answer)})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} sample answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
