"""Runs the eval harness against Vertex AI Search instead of the hand-built
pipeline - the Vertex AI Search benchmark comparison stretch goal.

Mirrors run_eval.py's conventions exactly (plain script, --config parsing,
tqdm, results/*.json report). --mode selects which Vertex config to run:
  search  - config A: Vertex Search API -> our own Generator/Judge/GroundingChecker
  answer  - config B: Vertex's own Answer API end-to-end
  both    - both, one after another (default)

Usage:
    python scripts/run_vertex_eval.py --mode search
    python scripts/run_vertex_eval.py --mode answer
    python scripts/run_vertex_eval.py --mode both
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
from rag_receipts.eval.runner import summarize
from rag_receipts.generation.generator import Generator
from rag_receipts.grounding.checker import GroundingChecker
from rag_receipts.vertex.answer_client import VertexAnswerer
from rag_receipts.vertex.eval_runner import run_vertex_eval_answer, run_vertex_eval_search
from rag_receipts.vertex.search_client import VertexRetriever


def _run_search_mode(cfg, questions) -> None:
    index_dir = Path(cfg.indexing.output.index_dir)
    metadata_path = index_dir / cfg.indexing.output.metadata_filename

    vertex_retriever = VertexRetriever.from_config(cfg.vertex, metadata_path)
    generator = Generator.from_config(cfg.generation)
    judge = Judge.from_config(cfg.eval)
    grounding_checker = GroundingChecker.from_config(cfg.grounding)

    results = []
    for question in tqdm(questions, desc="Running Vertex Search eval (config A)"):
        results.extend(
            run_vertex_eval_search(
                [question],
                vertex_retriever=vertex_retriever,
                generator=generator,
                judge=judge,
                grounding_checker=grounding_checker,
            )
        )

    summary = summarize(results)
    output_path = Path(cfg.vertex.search_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"summary": asdict(summary), "results": [asdict(r) for r in results]}
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[config A - Vertex Search] Wrote {len(results)} results to {output_path}")
    print(f"Questions: {summary.question_count}  Errors: {summary.error_count}")
    print(
        f"Retrieval  - precision: {summary.retrieval['mean_precision']:.3f}  "
        f"recall: {summary.retrieval['mean_recall']:.3f}  "
        f"MRR: {summary.retrieval['mean_mrr']:.3f}  "
        f"hit rate: {summary.retrieval['hit_rate']:.3f}"
    )
    print(f"Correctness accuracy: {summary.correctness_accuracy:.3f}")
    print(
        f"Grounding - grounded: {summary.grounding['grounded_rate']:.3f}  "
        f"contradicted: {summary.grounding['contradicted_rate']:.3f}"
    )


def _run_answer_mode(cfg, questions) -> None:
    index_dir = Path(cfg.indexing.output.index_dir)
    metadata_path = index_dir / cfg.indexing.output.metadata_filename

    vertex_answerer = VertexAnswerer.from_config(cfg.vertex, metadata_path)
    judge = Judge.from_config(cfg.eval)
    grounding_checker = GroundingChecker.from_config(cfg.grounding)

    results = []
    for question in tqdm(questions, desc="Running Vertex Answer eval (config B)"):
        results.extend(
            run_vertex_eval_answer(
                [question],
                vertex_answerer=vertex_answerer,
                judge=judge,
                grounding_checker=grounding_checker,
            )
        )

    summary = summarize(results)
    output_path = Path(cfg.vertex.answer_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_grounding_scores = [r.vertex_grounding_score for r in results if r.vertex_grounding_score is not None]
    mean_vertex_grounding_score = (
        sum(scored_grounding_scores) / len(scored_grounding_scores) if scored_grounding_scores else None
    )
    report = {
        "summary": asdict(summary),
        "mean_vertex_grounding_score": mean_vertex_grounding_score,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[config B - Vertex Answer API] Wrote {len(results)} results to {output_path}")
    print(f"Questions: {summary.question_count}  Errors: {summary.error_count}")
    print(
        "Retrieval (scored from cited chunk_ids only, not a full ranked search result list) - "
        f"precision: {summary.retrieval['mean_precision']:.3f}  "
        f"recall: {summary.retrieval['mean_recall']:.3f}  "
        f"MRR: {summary.retrieval['mean_mrr']:.3f}  "
        f"hit rate: {summary.retrieval['hit_rate']:.3f}"
    )
    print(f"Correctness accuracy: {summary.correctness_accuracy:.3f}")
    print(
        f"Grounding (our checker) - grounded: {summary.grounding['grounded_rate']:.3f}  "
        f"contradicted: {summary.grounding['contradicted_rate']:.3f}"
    )
    if mean_vertex_grounding_score is not None:
        print(f"Vertex's own self-reported grounding_score (mean): {mean_vertex_grounding_score:.3f}")


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    mode = "both"
    if "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
    if mode not in ("search", "answer", "both"):
        raise SystemExit(f"--mode must be one of search|answer|both, got {mode!r}")

    cfg = load_config(config_path)
    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    if mode in ("search", "both"):
        _run_search_mode(cfg, questions)
    if mode in ("answer", "both"):
        _run_answer_mode(cfg, questions)


if __name__ == "__main__":
    main()
