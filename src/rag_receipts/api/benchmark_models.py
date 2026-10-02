"""Condensed response schemas for GET /benchmarks (Stretch: demo benchmarks panel).

Thin Pydantic reshapings of the committed results/*.json files, mirroring the
QueryResponse/TimingOut convention in models.py - kept in a separate module
since models.py is scoped tightly to /query's request/response shape and this
payload has many more (small, flat) nested classes.

Every top-level section on BenchmarksResponse is Optional: benchmarks.py's
loader parses each source file independently and substitutes None on any
failure, so one missing/malformed file only blanks its own section rather
than failing the whole route.
"""

from __future__ import annotations

from pydantic import BaseModel


class StatsOut(BaseModel):
    p50: float
    p95: float
    mean: float
    count: int


class RetrievalMetricsOut(BaseModel):
    mean_precision: float
    mean_recall: float
    mean_mrr: float
    hit_rate: float
    count: int | None = None  # absent in vertex_comparison.json's per-config blocks


class CorrectnessTypeMetricOut(BaseModel):
    accuracy: float
    count: int


class GroundingMetricsOut(BaseModel):
    grounded_rate: float
    contradicted_rate: float
    ungrounded_rate: float
    fully_grounded_answer_rate: float
    citation_count: int | None = None  # absent in vertex_comparison.json's per-config blocks
    answer_count: int | None = None


class EvalSummaryOut(BaseModel):
    retrieval: RetrievalMetricsOut
    retrieval_by_type: dict[str, RetrievalMetricsOut]
    correctness_accuracy: float
    correctness_by_type: dict[str, CorrectnessTypeMetricOut]
    correctness_label_counts: dict[str, int]
    grounding: GroundingMetricsOut


class LatencyReportOut(BaseModel):
    stages: dict[str, StatsOut]


class RerankerSweepEntryOut(BaseModel):
    model_name: str
    retrieval: RetrievalMetricsOut
    rerank_latency: StatsOut
    total_retrieval_latency: StatsOut


class VertexGroundingOut(BaseModel):
    baseline: GroundingMetricsOut
    config_a_vertex_search: GroundingMetricsOut
    config_b_vertex_answer: GroundingMetricsOut
    config_b_vertex_self_reported_grounding_score: float


class VertexLatencyOut(BaseModel):
    baseline_staged: dict[str, StatsOut]
    config_a_vertex_search_end_to_end: StatsOut
    config_b_vertex_answer_end_to_end: StatsOut


class VertexComparisonOut(BaseModel):
    methodology_notes: dict[str, str]
    retrieval: dict[str, RetrievalMetricsOut]  # keys: baseline / config_a_vertex_search / config_b_vertex_answer
    correctness_accuracy: dict[str, float]
    grounding: VertexGroundingOut
    latency: VertexLatencyOut


class GenerationModelSummaryOut(BaseModel):
    model_id: str
    correctness_accuracy: float
    grounded_rate: float
    cost_usd_total: float
    latency_p50_s: float
    latency_p95_s: float


class GenerationComparisonOut(BaseModel):
    methodology_notes: dict[str, str]
    models: list[GenerationModelSummaryOut]


class PgvectorLatencySummaryOut(BaseModel):
    dense_search_p50: float
    retrieval_total_p50: float
    end_to_end_p50: float


class PgvectorComparisonOut(BaseModel):
    questions_checked: int
    set_match_count: int
    order_match_count: int
    max_score_diff_overall: float
    faiss_latency: PgvectorLatencySummaryOut
    pgvector_latency: PgvectorLatencySummaryOut


class BenchmarksResponse(BaseModel):
    eval_summary: EvalSummaryOut | None = None
    latency: LatencyReportOut | None = None
    reranker_sweep: list[RerankerSweepEntryOut] | None = None
    vertex_comparison: VertexComparisonOut | None = None
    generation_comparison: GenerationComparisonOut | None = None
    pgvector_comparison: PgvectorComparisonOut | None = None
