"""HTTP service over the retrieval + generation pipeline (Step 8).

Deliberately thin: Retriever (Step 3) and Generator (Step 4) are already
designed as load-once/query-cheap components (see their own module
docstrings), so this module's only job is wiring - load both once at
startup, expose them over a handful of routes, and translate their existing
result dataclasses into the HTTP response shape. No retrieval/generation/
citation logic lives here. GET /benchmarks follows the same load-once
pattern for the committed results/*.json files (see benchmarks.py).

create_app() takes retriever/generator directly so tests can inject fakes
and skip the real model-loading path in `lifespan` entirely - `app`
(module-level, for uvicorn) uses the real pipeline via config.yaml + GCS.

Access control: the deployed service is public (`--allow-unauthenticated`),
gated by a shared secret (DEMO_API_KEY, see _check_demo_key) rather than
per-user auth - native Cloud Run IAP was tried first but requires the GCP
project to belong to an Organization (this account has none), so a shared
demo key is the fallback that still lets a recruiter self-serve the live URL
without per-account setup.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from rag_receipts.api.benchmark_models import BenchmarksResponse
from rag_receipts.api.benchmarks import load_benchmarks
from rag_receipts.api.models import CitationOut, QueryRequest, QueryResponse, TimingOut
from rag_receipts.api.rate_limit import RateLimiter
from rag_receipts.api.startup import ensure_index_artifacts
from rag_receipts.config import load_config
from rag_receipts.generation.generator import Generator
from rag_receipts.generation.models import GeneratedAnswer
from rag_receipts.retrieval.models import RetrievedChunk
from rag_receipts.retrieval.pipeline import Retriever
from rag_receipts.telemetry.models import RetrievalTiming
from rag_receipts.telemetry.timing import Stopwatch

logger = logging.getLogger("rag_receipts.api")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = float(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "300"))


class RetrieverLike(Protocol):
    def retrieve_with_timing(
        self, query: str
    ) -> tuple[list[RetrievedChunk], RetrievalTiming]: ...


class GeneratorLike(Protocol):
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer: ...


def _extract_identity(request: Request) -> str:
    """Rate-limiter key. The service is gated by a shared demo key (see
    _check_demo_key below), not per-user auth, so client IP is the best
    identity signal available - not perfect (a shared corporate NAT could
    lump distinct recruiters together), but good enough to stop one caller
    from hammering the real Anthropic API."""
    return request.client.host if request.client else "unknown"


def _check_demo_key(request: Request) -> None:
    """Shared-secret gate for /query (Secret Manager -> DEMO_API_KEY env var
    at deploy time). Deliberately reads the env var per-call rather than at
    import time so it's test-controllable via monkeypatch, and so local dev
    with no DEMO_API_KEY set stays fully open (same no-op-when-unset
    convention as ensure_index_artifacts' INDEX_GCS_BUCKET)."""
    expected = os.environ.get("DEMO_API_KEY")
    if not expected:
        return
    if request.headers.get("X-Demo-Key") != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-Demo-Key header")


def create_app(
    *,
    config_path: str | Path = "config/config.yaml",
    static_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    retriever: RetrieverLike | None = None,
    generator: GeneratorLike | None = None,
    rate_limiter: RateLimiter | None = None,
    benchmarks: BenchmarksResponse | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.retriever is None or app.state.generator is None:
            cfg = load_config(app.state.config_path)
            ensure_index_artifacts(cfg)
            app.state.retriever = app.state.retriever or Retriever.from_config(cfg)
            app.state.generator = app.state.generator or Generator.from_config(cfg.generation)
        if app.state.benchmarks is None:
            app.state.benchmarks = load_benchmarks(app.state.results_dir)
        app.state.ready = True
        yield

    app = FastAPI(title="RAG With Receipts", lifespan=lifespan)
    app.state.config_path = config_path
    app.state.static_dir = Path(static_dir or os.environ.get("STATIC_DIR", "static"))
    app.state.results_dir = Path(results_dir or os.environ.get("RESULTS_DIR", "results"))
    app.state.retriever = retriever
    app.state.generator = generator
    app.state.benchmarks = benchmarks
    app.state.ready = False
    app.state.rate_limiter = rate_limiter or RateLimiter(
        max_requests=RATE_LIMIT_MAX_REQUESTS, window_seconds=RATE_LIMIT_WINDOW_SECONDS
    )

    @app.get("/livez")
    def livez() -> dict:
        # Not /healthz: confirmed live against the deployed Cloud Run service
        # that the literal path "/healthz" is intercepted and 404'd by
        # Google's infrastructure before it ever reaches the container
        # (every other path, including ones containing "health", reaches the
        # app fine - verified by comparing response headers: the intercepted
        # response has no `server: Google Frontend` / `x-cloud-trace-context`
        # that every real app response carries).
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request) -> dict:
        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ready"}

    @app.get("/")
    def index(request: Request) -> FileResponse:
        path = request.app.state.static_dir / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="demo page not found")
        return FileResponse(path)

    @app.get("/benchmarks", response_model=BenchmarksResponse)
    def benchmarks(request: Request) -> BenchmarksResponse:
        # No demo-key gate, no rate limit: static local JSON, no external API
        # cost. Never 404s/500s on its own - a missing/malformed results file
        # just leaves that one section None (see benchmarks.py's loader).
        return request.app.state.benchmarks

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest, request: Request) -> QueryResponse:
        _check_demo_key(request)

        if not request.app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")

        identity = _extract_identity(request)
        if not request.app.state.rate_limiter.allow(identity):
            raise HTTPException(
                status_code=429, detail="rate limit exceeded - try again later"
            )

        retriever: RetrieverLike = request.app.state.retriever
        generator: GeneratorLike = request.app.state.generator

        chunks, retrieval_timing = retriever.retrieve_with_timing(req.query)
        with Stopwatch() as generate_sw:
            try:
                answer = generator.generate(req.query, chunks)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=502, detail=f"generation failed: {exc}"
                ) from exc

        timing = TimingOut(
            embed_query_s=retrieval_timing.embed_query_s,
            dense_search_s=retrieval_timing.dense_search_s,
            rerank_s=retrieval_timing.rerank_s,
            generate_s=generate_sw.elapsed_seconds,
            total_s=retrieval_timing.total_s + generate_sw.elapsed_seconds,
        )

        logger.info(
            json.dumps(
                {
                    "event": "query",
                    "identity": identity,
                    "answerable": answer.answerable,
                    "citation_count": len(answer.citations),
                    "hallucinated_citation_count": len(answer.hallucinated_citation_ids),
                    "timing": timing.model_dump(),
                }
            )
        )

        return QueryResponse(
            query=req.query,
            answerable=answer.answerable,
            answer=answer.answer,
            citations=[
                CitationOut(
                    chunk_id=c.chunk_id,
                    claim=c.claim,
                    page_title=c.chunk.page_title,
                    url=c.chunk.url,
                    section_path=c.chunk.section_path,
                    heading_anchor=c.chunk.heading_anchor,
                )
                for c in answer.citations
            ],
            hallucinated_citation_ids=answer.hallucinated_citation_ids,
            timing=timing,
        )

    return app


app = create_app()
