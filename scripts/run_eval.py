"""Run the eval harness: retrieval precision + LLM-as-judge correctness over the gold
Q/A set at config.eval.dataset_path, writing a results report to config.eval.output_path
(default results/eval_report.json) - Step 5's "results report" deliverable.

Plain script (no typer), matching sample_answers.py/sample_retrievals.py's convention -
this is a batch run over a dataset file, not a single interactive query.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --config config/config.yaml
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.eval.judge import Judge
from rag_receipts.eval.runner import run_eval, summarize
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.pipeline import Retriever


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    cfg = load_config(config_path)

    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)
    judge = Judge.from_config(cfg.eval)

    results = []
    for question in tqdm(questions, desc="Running eval"):
        results.extend(
            run_eval([question], retriever=retriever, generator=generator, judge=judge)
        )

    summary = summarize(results)

    output_path = Path(cfg.eval.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": asdict(summary), "results": [asdict(r) for r in results]}
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {len(results)} results to {output_path}\n")
    print(f"Questions: {summary.question_count}  Errors: {summary.error_count}")
    print(
        f"Retrieval  - precision: {summary.retrieval['mean_precision']:.3f}  "
        f"recall: {summary.retrieval['mean_recall']:.3f}  "
        f"MRR: {summary.retrieval['mean_mrr']:.3f}  "
        f"hit rate: {summary.retrieval['hit_rate']:.3f}"
    )
    print(f"Correctness accuracy: {summary.correctness_accuracy:.3f}")
    for qtype, stats in summary.correctness_by_type.items():
        print(f"  {qtype}: {stats['accuracy']:.3f} (n={stats['count']})")
    print(f"Label counts: {summary.correctness_label_counts}")


if __name__ == "__main__":
    main()
