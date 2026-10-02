"""Step 7's measured optimization: a reranker sweep comparing retrieval precision
against p95 latency across the current baseline (bge-reranker-base), a stronger
variant (bge-reranker-v2-m3), and a faster variant (ms-marco-MiniLM-L-6-v2).

Retrieval-only - no Claude generation calls. Swapping the reranker only changes
which chunks are retrieved, which Step 5's retrieval metrics (precision/recall/
mrr/hit-rate) already measure without a generation call; generation latency is
reranker-independent (same top_k_final chunks in, regardless of which reranker
picked them), so running it 3x here would add real API cost for no signal.

The FAISS index, metadata, and embedding model are reranker-independent and are
loaded once, outside the candidate loop - only the reranker itself is rebuilt per
candidate.

Writes results/reranker_sweep.json: one record per candidate with retrieval
precision/recall/mrr/hit-rate (over the 50 answerable gold questions) and rerank-
stage + total-retrieval p50/p95/mean latency (over all 55).

Usage:
    python scripts/sweep_reranker.py
    python scripts/sweep_reranker.py --config config/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.eval.retrieval_metrics import aggregate_retrieval_scores, score_retrieval
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.retrieval.config import RerankerConfig
from rag_receipts.retrieval.pipeline import run_retrieval_with_timing
from rag_receipts.retrieval.reranker import Reranker
from rag_receipts.telemetry.timing import summarize_durations
from rag_receipts.vectorstore.faiss_store import FaissVectorStore

CANDIDATES = [
    "BAAI/bge-reranker-base",  # current default/baseline
    "BAAI/bge-reranker-v2-m3",  # stronger
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # faster
]


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    cfg = load_config(config_path)

    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    vector_store = FaissVectorStore.from_config(cfg.indexing)
    embedder = Embedder.from_config(cfg.indexing.embedding)

    records = []
    for model_name in CANDIDATES:
        print(f"\n--- {model_name} ---")
        reranker_config = RerankerConfig(
            model_name=model_name,
            batch_size=cfg.retrieval.reranker.batch_size,
            device=cfg.retrieval.reranker.device,
            show_progress_bar=cfg.retrieval.reranker.show_progress_bar,
        )
        reranker = Reranker.from_config(reranker_config)

        retrieval_scores = []
        rerank_seconds = []
        total_seconds = []
        for question in tqdm(questions, desc=model_name):
            chunks, timing = run_retrieval_with_timing(
                question.question,
                vector_store=vector_store,
                embedder=embedder,
                reranker=reranker,
                config=cfg.retrieval,
            )
            rerank_seconds.append(timing.rerank_s)
            total_seconds.append(timing.total_s)
            if question.answerable:
                retrieval_scores.append(score_retrieval(chunks, question.gold_chunk_ids))

        record = {
            "model_name": model_name,
            "retrieval": aggregate_retrieval_scores(retrieval_scores),
            "rerank_latency": summarize_durations(rerank_seconds),
            "total_retrieval_latency": summarize_durations(total_seconds),
        }
        records.append(record)

    output_path = Path("results/reranker_sweep.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote reranker sweep to {output_path}\n")
    print(f"{'model':40s} {'recall':>7s} {'hit_rate':>9s} {'rerank_p95':>11s} {'total_p95':>10s}")
    for r in records:
        print(
            f"{r['model_name']:40s} "
            f"{r['retrieval']['mean_recall']:7.3f} "
            f"{r['retrieval']['hit_rate']:9.3f} "
            f"{r['rerank_latency']['p95'] * 1000:9.1f}ms "
            f"{r['total_retrieval_latency']['p95'] * 1000:8.1f}ms"
        )


if __name__ == "__main__":
    main()
