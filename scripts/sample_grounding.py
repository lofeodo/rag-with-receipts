"""Run a handful of hand-picked queries end-to-end (retrieval + generation + grounding
check) and write results/sample_grounding.json - a committed, small visibility artifact
showing per-citation grounded/ungrounded/contradicted verdicts without requiring the
reader to run the pipeline themselves (mirrors Step 4's sample_answers.py).

Duplicates sample_answers.py's QUERIES list for the same reason sample_answers.py
duplicates sample_retrievals.py's: scripts/ has no __init__.py, so a
`python scripts/sample_grounding.py` invocation puts scripts/ itself, not the repo root,
on sys.path, and a cross-script import doesn't reliably resolve.

Usage:
    python scripts/sample_grounding.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rag_receipts.config import load_config
from rag_receipts.generation.generator import Generator
from rag_receipts.grounding.checker import GroundingChecker, summarize_grounding
from rag_receipts.retrieval.pipeline import Retriever

QUERIES = [
    "What items are required to start Monkey Madness I?",
    "How much Slayer experience is needed for level 70?",
    "What is the Abyssal whip's special attack?",
    "What potions require Herblore level 78?",
    "How do you defeat Zooknock in Monkey Madness I?",
    "What are the steps to complete the Cook's Assistant quest?",  # out of corpus scope
]

OUTPUT_PATH = Path("results/sample_grounding.json")


def main() -> None:
    cfg = load_config("config/config.yaml")
    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)
    grounding_checker = GroundingChecker.from_config(cfg.grounding)

    samples = []
    for query in QUERIES:
        chunks = retriever.retrieve(query)
        answer = generator.generate(query, chunks)
        grounding = summarize_grounding(grounding_checker.check(answer.citations))
        samples.append(
            {
                "query": query,
                "answer": answer.answer,
                "answerable": answer.answerable,
                "grounding": asdict(grounding),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(samples)} sample grounding checks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
