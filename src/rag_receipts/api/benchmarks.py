"""Loader for results/*.json -> condensed BenchmarksResponse (Stretch: demo
benchmarks panel).

Mirrors the Retriever/Generator load-once-at-startup pattern (see app.py's
lifespan): parses the committed results/*.json files once, extracts only the
condensed fields the demo panel needs (never the large per-question
`results`/`per_question` arrays), and returns a single typed payload.

Each section is parsed independently and defaults to None on any failure
(missing file, malformed JSON, unexpected shape) so one regenerated-but-not-
yet-committed results file can't break the whole /benchmarks route - the
panel just omits that one section. Failures are logged, never raised.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, TypeVar

from rag_receipts.api.benchmark_models import (
    BenchmarksResponse,
    EvalSummaryOut,
    GenerationComparisonOut,
    GenerationModelSummaryOut,
    LatencyReportOut,
    PgvectorComparisonOut,
    PgvectorLatencySummaryOut,
    RerankerSweepEntryOut,
    VertexComparisonOut,
)

logger = logging.getLogger("rag_receipts.api")

T = TypeVar("T")

_SAFE_EXCEPTIONS = (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _safe(fn: Callable[[Path], T], results_dir: Path, label: str) -> T | None:
    try:
        return fn(results_dir)
    except _SAFE_EXCEPTIONS as exc:
        logger.warning("benchmarks: failed to load %s: %s", label, exc)
        return None


def _load_eval_summary(results_dir: Path) -> EvalSummaryOut:
    summary = _read_json(results_dir / "eval_report.json")["summary"]
    return EvalSummaryOut(
        retrieval=summary["retrieval"],
        retrieval_by_type=summary["retrieval_by_type"],
        correctness_accuracy=summary["correctness_accuracy"],
        correctness_by_type=summary["correctness_by_type"],
        correctness_label_counts=summary["correctness_label_counts"],
        grounding=summary["grounding"],
    )


def _load_latency_report(results_dir: Path, filename: str = "latency_report.json") -> LatencyReportOut:
    data = _read_json(results_dir / filename)
    return LatencyReportOut(stages=data["stages"])


def _load_reranker_sweep(results_dir: Path) -> list[RerankerSweepEntryOut]:
    data = _read_json(results_dir / "reranker_sweep.json")
    return [RerankerSweepEntryOut(**entry) for entry in data]


def _load_vertex_comparison(results_dir: Path) -> VertexComparisonOut:
    data = _read_json(results_dir / "vertex_comparison.json")
    return VertexComparisonOut(
        methodology_notes=data["methodology_notes"],
        retrieval=data["retrieval"],
        correctness_accuracy=data["correctness_accuracy"],
        grounding=data["grounding"],
        latency=data["latency"],
    )


def _load_generation_comparison(results_dir: Path) -> GenerationComparisonOut:
    data = _read_json(results_dir / "generation_model_comparison.json")
    models = [
        GenerationModelSummaryOut(
            model_id=model_id,
            correctness_accuracy=model["eval_summary"]["correctness_accuracy"],
            grounded_rate=model["eval_summary"]["grounding"]["grounded_rate"],
            cost_usd_total=model["cost_usd"]["total"],
            latency_p50_s=model["latency_s"]["p50"],
            latency_p95_s=model["latency_s"]["p95"],
        )
        for model_id, model in data["models"].items()
    ]
    return GenerationComparisonOut(methodology_notes=data["methodology_notes"], models=models)


def _load_pgvector_comparison(results_dir: Path) -> PgvectorComparisonOut:
    equivalence = _read_json(results_dir / "pgvector_equivalence.json")
    faiss_stages = _read_json(results_dir / "latency_report_faiss.json")["stages"]
    pgvector_stages = _read_json(results_dir / "latency_report_pgvector.json")["stages"]

    def _latency_summary(stages: dict) -> PgvectorLatencySummaryOut:
        return PgvectorLatencySummaryOut(
            dense_search_p50=stages["dense_search_s"]["p50"],
            retrieval_total_p50=stages["retrieval_total_s"]["p50"],
            end_to_end_p50=stages["end_to_end_s"]["p50"],
        )

    return PgvectorComparisonOut(
        questions_checked=equivalence["questions_checked"],
        set_match_count=equivalence["set_match_count"],
        order_match_count=equivalence["order_match_count"],
        max_score_diff_overall=equivalence["max_score_diff_overall"],
        faiss_latency=_latency_summary(faiss_stages),
        pgvector_latency=_latency_summary(pgvector_stages),
    )


def load_benchmarks(results_dir: Path) -> BenchmarksResponse:
    return BenchmarksResponse(
        eval_summary=_safe(_load_eval_summary, results_dir, "eval_report.json"),
        latency=_safe(_load_latency_report, results_dir, "latency_report.json"),
        reranker_sweep=_safe(_load_reranker_sweep, results_dir, "reranker_sweep.json"),
        vertex_comparison=_safe(_load_vertex_comparison, results_dir, "vertex_comparison.json"),
        generation_comparison=_safe(
            _load_generation_comparison, results_dir, "generation_model_comparison.json"
        ),
        pgvector_comparison=_safe(
            _load_pgvector_comparison,
            results_dir,
            "pgvector_equivalence.json + latency_report_{faiss,pgvector}.json",
        ),
    )
