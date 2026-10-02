"""Dense search + cross-encoder rerank orchestration (Step 3).

Retriever separates expensive load-once state (vector store, embedding model,
reranker model) from cheap per-query retrieval, so Step 4/5/8 can construct
one Retriever and call .retrieve() in a loop instead of reloading models per
call. The vector store backend (FAISS by default, pgvector as a config-driven
swap - see CLAUDE.md's pgvector stretch goal status note) is selected once in
Retriever.from_config and is opaque to everything below that point - this
module only ever talks to the VectorStore Protocol, never to faiss/pg8000
directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_receipts.config import AppConfig
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.retrieval.config import RetrievalConfig
from rag_receipts.retrieval.models import RetrievedChunk
from rag_receipts.retrieval.reranker import Reranker, rerank_text
from rag_receipts.telemetry.models import RetrievalTiming
from rag_receipts.telemetry.timing import Stopwatch
from rag_receipts.vectorstore.base import VectorStore
from rag_receipts.vectorstore.faiss_store import FaissVectorStore


def build_vector_store(config: AppConfig) -> VectorStore:
    """Dispatch point for the pgvector stretch goal's config-driven swap.

    PgvectorStore is imported lazily so that the (optional, pgvector extra)
    Cloud SQL Connector dependency is never required for the default
    indexing.vector_index=faiss path.
    """
    if config.indexing.vector_index == "faiss":
        return FaissVectorStore.from_config(config.indexing)
    if config.indexing.vector_index == "pgvector":
        from rag_receipts.vectorstore.pgvector_store import PgvectorStore

        return PgvectorStore.from_config(config.indexing.pgvector)
    raise ValueError(
        f"Unknown indexing.vector_index={config.indexing.vector_index!r}; "
        f"expected 'faiss' or 'pgvector'"
    )


def run_retrieval(
    query: str,
    *,
    vector_store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    chunks, _timing = run_retrieval_with_timing(
        query,
        vector_store=vector_store,
        embedder=embedder,
        reranker=reranker,
        config=config,
    )
    return chunks


def run_retrieval_with_timing(
    query: str,
    *,
    vector_store: VectorStore,
    embedder: Embedder,
    reranker: Reranker,
    config: RetrievalConfig,
) -> tuple[list[RetrievedChunk], RetrievalTiming]:
    """Same as run_retrieval, but also returns per-stage wall-clock seconds
    (Step 7) - used by scripts/measure_latency.py and scripts/sweep_reranker.py.
    run_retrieval is a thin wrapper around this that discards the timing."""
    with Stopwatch() as embed_sw:
        query_vec = embedder.encode_queries([query])[0]

    with Stopwatch() as dense_sw:
        hits = vector_store.search(query_vec, config.top_k_dense)

    if not hits:
        timing = RetrievalTiming(
            embed_query_s=embed_sw.elapsed_seconds,
            dense_search_s=dense_sw.elapsed_seconds,
            rerank_s=0.0,
            total_s=embed_sw.elapsed_seconds + dense_sw.elapsed_seconds,
        )
        return [], timing

    passages = [rerank_text(hit.page_title, hit.section_path, hit.text) for hit in hits]
    with Stopwatch() as rerank_sw:
        rerank_scores = reranker.score(query, passages)

    candidates = [
        RetrievedChunk(
            chunk_id=hit.chunk_id,
            page_title=hit.page_title,
            url=hit.url,
            section_path=hit.section_path,
            heading_anchor=hit.heading_anchor,
            source_type=hit.source_type,
            text=hit.text,
            token_count=hit.token_count,
            dense_score=hit.score,
            rerank_score=float(rerank_score),
            dense_rank=dense_rank,
            final_rank=0,
        )
        for dense_rank, (hit, rerank_score) in enumerate(zip(hits, rerank_scores), start=1)
    ]

    candidates.sort(key=lambda c: c.rerank_score, reverse=True)
    top = candidates[: config.top_k_final]
    for rank, chunk in enumerate(top, start=1):
        chunk.final_rank = rank

    timing = RetrievalTiming(
        embed_query_s=embed_sw.elapsed_seconds,
        dense_search_s=dense_sw.elapsed_seconds,
        rerank_s=rerank_sw.elapsed_seconds,
        total_s=embed_sw.elapsed_seconds + dense_sw.elapsed_seconds + rerank_sw.elapsed_seconds,
    )
    return top, timing


@dataclass
class Retriever:
    """Owns all expensive load-once state. Construct once via from_config;
    .retrieve() does zero I/O or model loading per call."""

    config: RetrievalConfig
    vector_store: VectorStore
    embedder: Embedder
    reranker: Reranker

    @classmethod
    def from_config(cls, config: AppConfig) -> "Retriever":
        vector_store = build_vector_store(config)
        embedder = Embedder.from_config(config.indexing.embedding)
        reranker = Reranker.from_config(config.retrieval.reranker)
        return cls(
            config=config.retrieval,
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
        )

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return run_retrieval(
            query,
            vector_store=self.vector_store,
            embedder=self.embedder,
            reranker=self.reranker,
            config=self.config,
        )

    def retrieve_with_timing(self, query: str) -> tuple[list[RetrievedChunk], RetrievalTiming]:
        return run_retrieval_with_timing(
            query,
            vector_store=self.vector_store,
            embedder=self.embedder,
            reranker=self.reranker,
            config=self.config,
        )
