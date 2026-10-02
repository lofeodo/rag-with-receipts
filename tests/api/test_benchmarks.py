from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from rag_receipts.api.app import create_app
from rag_receipts.api.benchmark_models import BenchmarksResponse
from rag_receipts.api.benchmarks import load_benchmarks

from .helpers import FakeGenerator, FakeRetriever

FIXTURES_DIR = Path(__file__).parent / "fixtures"

FIXTURE_FILES = [
    "eval_report.json",
    "latency_report.json",
    "reranker_sweep.json",
    "vertex_comparison.json",
    "generation_model_comparison.json",
    "pgvector_equivalence.json",
    "latency_report_faiss.json",
    "latency_report_pgvector.json",
]


def _copy_fixtures(dest: Path, *, skip: str | None = None, corrupt: str | None = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in FIXTURE_FILES:
        if name == skip:
            continue
        if name == corrupt:
            (dest / name).write_text("{not valid json", encoding="utf-8")
            continue
        shutil.copy(FIXTURES_DIR / name, dest / name)


def test_load_benchmarks_all_sections_present(tmp_path):
    _copy_fixtures(tmp_path)

    result = load_benchmarks(tmp_path)

    assert result.eval_summary is not None
    assert result.eval_summary.retrieval.hit_rate == 0.95
    assert result.latency is not None
    assert result.latency.stages["generate_s"].p50 == 2.9
    assert result.reranker_sweep is not None
    assert [m.model_name for m in result.reranker_sweep] == ["fixture-base", "fixture-fast"]
    assert result.vertex_comparison is not None
    assert result.vertex_comparison.correctness_accuracy["baseline"] == 0.84
    assert result.generation_comparison is not None
    assert [m.model_id for m in result.generation_comparison.models] == [
        "claude-sonnet-5",
        "claude-haiku-4-5",
    ]
    assert result.pgvector_comparison is not None
    assert result.pgvector_comparison.set_match_count == 2
    assert result.pgvector_comparison.faiss_latency.dense_search_p50 == 0.0013
    assert result.pgvector_comparison.pgvector_latency.dense_search_p50 == 0.098


def test_load_benchmarks_missing_file_omits_only_that_section(tmp_path):
    _copy_fixtures(tmp_path, skip="vertex_comparison.json")

    result = load_benchmarks(tmp_path)

    assert result.vertex_comparison is None
    assert result.eval_summary is not None
    assert result.latency is not None
    assert result.reranker_sweep is not None
    assert result.generation_comparison is not None
    assert result.pgvector_comparison is not None


def test_load_benchmarks_malformed_json_omits_only_that_section(tmp_path):
    _copy_fixtures(tmp_path, corrupt="reranker_sweep.json")

    result = load_benchmarks(tmp_path)

    assert result.reranker_sweep is None
    assert result.eval_summary is not None
    assert result.vertex_comparison is not None


def test_load_benchmarks_pgvector_section_requires_all_three_files(tmp_path):
    _copy_fixtures(tmp_path, skip="latency_report_pgvector.json")

    result = load_benchmarks(tmp_path)

    assert result.pgvector_comparison is None
    # Confirms this is specific to the 3-file pgvector bundle, not a global failure.
    assert result.eval_summary is not None


def test_load_benchmarks_empty_dir_returns_all_none_sections(tmp_path):
    result = load_benchmarks(tmp_path)

    assert result == BenchmarksResponse()


def _app(**overrides):
    kwargs = {"retriever": FakeRetriever(), "generator": FakeGenerator()}
    kwargs.update(overrides)
    return create_app(**kwargs)


def test_benchmarks_route_200_with_injected_payload():
    payload = load_benchmarks(FIXTURES_DIR)
    app = _app(benchmarks=payload)
    client = TestClient(app)

    resp = client.get("/benchmarks")

    assert resp.status_code == 200
    body = resp.json()
    assert body["eval_summary"]["retrieval"]["hit_rate"] == 0.95
    assert body["reranker_sweep"][1]["model_name"] == "fixture-fast"


def test_benchmarks_route_200_with_all_sections_none_when_results_dir_empty(tmp_path):
    app = _app(results_dir=tmp_path)
    with TestClient(app) as client:
        resp = client.get("/benchmarks")

    assert resp.status_code == 200
    body = resp.json()
    assert all(body[key] is None for key in body)


def test_benchmarks_route_loads_from_results_dir_at_startup(tmp_path):
    _copy_fixtures(tmp_path)
    app = _app(results_dir=tmp_path)

    with TestClient(app) as client:
        resp = client.get("/benchmarks")

    assert resp.status_code == 200
    body = resp.json()
    assert body["eval_summary"]["retrieval"]["hit_rate"] == 0.95
    assert body["pgvector_comparison"]["set_match_count"] == 2
