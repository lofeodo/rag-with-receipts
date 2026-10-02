"""Latency instrumentation (Step 7): per-stage p50/p95/mean over the hot path -
query embedding, dense/vector-store search, cross-encoder reranking, and Claude
generation - run against the real 55-question gold set, writing
results/latency_report.json (or results/latency_report_{vector_index}.json
when --vector-index overrides the configured backend).

Model loading (Retriever.from_config / Generator.from_config) happens once, outside
the timed loop, and is deliberately excluded from the per-stage stats - it's a
cold-start concern for Step 8's deploy, not a per-query latency number.

The reranker-choice question ("which model, at what latency/precision tradeoff") is
NOT answered here - that's scripts/sweep_reranker.py's job. This script measures the
pipeline as currently configured in config.yaml (or as overridden by --vector-index).

--vector-index {faiss,pgvector} (pgvector stretch goal): overrides
config.indexing.vector_index before constructing Retriever, so the exact same
measurement code path runs against both backends - running literally the same
code, not a separate script per backend, is itself part of what makes the
faiss-vs-pgvector latency comparison apples-to-apples. Omitting the flag keeps
today's default behavior (output path, backend) byte-for-byte unchanged.

Usage:
    python scripts/measure_latency.py
    python scripts/measure_latency.py --config config/config.yaml
    python scripts/measure_latency.py --vector-index pgvector
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.generation.generator import Generator
from rag_receipts.retrieval.pipeline import Retriever
from rag_receipts.telemetry.timing import summarize_durations


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    vector_index_override = None
    if "--vector-index" in sys.argv:
        vector_index_override = sys.argv[sys.argv.index("--vector-index") + 1]

    cfg = load_config(config_path)
    if vector_index_override is not None:
        cfg.indexing.vector_index = vector_index_override

    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    retriever = Retriever.from_config(cfg)
    generator = Generator.from_config(cfg.generation)

    stage_durations: dict[str, list[float]] = {
        "embed_query_s": [],
        "dense_search_s": [],
        "rerank_s": [],
        "retrieval_total_s": [],
        "generate_s": [],
        "end_to_end_s": [],
    }
    errors: list[str] = []

    for question in tqdm(questions, desc="Measuring latency"):
        chunks, timing = retriever.retrieve_with_timing(question.question)

        t0 = time.perf_counter()
        try:
            generator.generate(question.question, chunks)
        except RuntimeError as exc:
            errors.append(f"{question.id}: {exc}")
            continue
        generate_s = time.perf_counter() - t0

        stage_durations["embed_query_s"].append(timing.embed_query_s)
        stage_durations["dense_search_s"].append(timing.dense_search_s)
        stage_durations["rerank_s"].append(timing.rerank_s)
        stage_durations["retrieval_total_s"].append(timing.total_s)
        stage_durations["generate_s"].append(generate_s)
        stage_durations["end_to_end_s"].append(timing.total_s + generate_s)

    summary = {stage: summarize_durations(durations) for stage, durations in stage_durations.items()}
    report = {
        "question_count": len(questions),
        "error_count": len(errors),
        "errors": errors,
        "stages": summary,
    }

    output_path = Path(
        f"results/latency_report_{cfg.indexing.vector_index}.json"
        if vector_index_override is not None
        else "results/latency_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote latency report to {output_path}\n")
    print(f"Questions: {len(questions)}  Errors: {len(errors)}")
    for stage, stats in summary.items():
        print(
            f"  {stage:20s} p50={stats['p50'] * 1000:7.1f}ms  p95={stats['p95'] * 1000:7.1f}ms  "
            f"mean={stats['mean'] * 1000:7.1f}ms  n={stats['count']}"
        )


if __name__ == "__main__":
    main()
