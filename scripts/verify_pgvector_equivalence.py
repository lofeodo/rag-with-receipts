"""pgvector-on-Cloud-SQL equivalence verification (pgvector stretch goal).

NOT a retrieval-quality benchmark - see CLAUDE.md's pgvector stretch goal
status note for why one isn't meaningful here. Exact nearest-neighbor search
over the same embeddings is mathematically guaranteed to retrieve the same
chunks regardless of backend (FAISS IndexFlatIP vs. pgvector's `<=>` with no
ivfflat/hnsw index), modulo floating-point tie noise. This script checks that
guarantee actually holds against the real corpus and the real Cloud SQL
instance, at the dense-search stage specifically (config.retrieval.top_k_dense,
not the post-rerank top_k_final) - the dense stage is the actual VectorStore
contract surface; any ordering noise introduced there would otherwise be
invisible after reranking.

Reuses the real 55 data/eval/qa_pairs.json questions (via load_eval_questions)
rather than a synthetic sample - the only hand-authored, domain-realistic
query set this project has.

For each question: set equality of returned chunk_ids is the pass/fail
criterion (any mismatch is unconditionally a failure - a wrong distance
direction, stale table contents, or similar bug). List-order equality is a
secondary signal; an order mismatch is only acceptable when every swapped
pair's score gap is below a small epsilon (a tie-flip from float32-vs-
Postgres-float arithmetic noise, not a real divergence).

Usage:
    python scripts/verify_pgvector_equivalence.py
    python scripts/verify_pgvector_equivalence.py --config config/config.yaml
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tqdm import tqdm

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.vectorstore.faiss_store import FaissVectorStore
from rag_receipts.vectorstore.pgvector_store import PgvectorStore

TIE_FLIP_EPSILON = 1e-4


def _classify_order_mismatch(
    faiss_ids: list[str], pg_ids: list[str], faiss_scores: dict[str, float], pg_scores: dict[str, float]
) -> str:
    """Every id present in both lists but at different positions - if every such
    id's FAISS/pgvector score gap is within epsilon, it's a tie-flip (noise);
    otherwise it's a real divergence (a bug)."""
    max_gap = 0.0
    for cid in faiss_ids:
        if cid in pg_scores:
            gap = abs(faiss_scores[cid] - pg_scores[cid])
            max_gap = max(max_gap, gap)
    return "tie_flip" if max_gap < TIE_FLIP_EPSILON else "real_divergence"


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    cfg = load_config(config_path)

    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    faiss_store = FaissVectorStore.from_config(cfg.indexing)
    pg_store = PgvectorStore.from_config(cfg.indexing.pgvector)
    embedder = Embedder.from_config(cfg.indexing.embedding)

    top_k = cfg.retrieval.top_k_dense
    per_question = []
    set_match_count = 0
    order_match_count = 0
    tie_flip_count = 0
    real_divergence_count = 0
    max_score_diff_overall = 0.0

    try:
        for question in tqdm(questions, desc="Verifying equivalence"):
            query_vec = embedder.encode_queries([question.question])[0]

            faiss_hits = faiss_store.search(query_vec, top_k)
            pg_hits = pg_store.search(query_vec, top_k)

            faiss_ids = [h.chunk_id for h in faiss_hits]
            pg_ids = [h.chunk_id for h in pg_hits]
            faiss_scores = {h.chunk_id: h.score for h in faiss_hits}
            pg_scores = {h.chunk_id: h.score for h in pg_hits}

            set_match = set(faiss_ids) == set(pg_ids)
            order_match = faiss_ids == pg_ids

            common_ids = set(faiss_ids) & set(pg_ids)
            max_score_diff = (
                max(abs(faiss_scores[cid] - pg_scores[cid]) for cid in common_ids)
                if common_ids
                else 0.0
            )
            max_score_diff_overall = max(max_score_diff_overall, max_score_diff)

            order_mismatch_type = None
            if set_match and not order_match:
                order_mismatch_type = _classify_order_mismatch(
                    faiss_ids, pg_ids, faiss_scores, pg_scores
                )
                if order_mismatch_type == "tie_flip":
                    tie_flip_count += 1
                else:
                    real_divergence_count += 1

            if set_match:
                set_match_count += 1
            if order_match:
                order_match_count += 1

            per_question.append(
                {
                    "id": question.id,
                    "set_match": set_match,
                    "order_match": order_match,
                    "max_score_diff": max_score_diff,
                    "order_mismatch_type": order_mismatch_type,
                }
            )
    finally:
        pg_store.close()

    report = {
        "questions_checked": len(questions),
        "top_k": top_k,
        "set_match_count": set_match_count,
        "order_match_count": order_match_count,
        "tie_flip_count": tie_flip_count,
        "real_divergence_count": real_divergence_count,
        "max_score_diff_overall": max_score_diff_overall,
        "per_question": per_question,
    }

    output_path = Path("results/pgvector_equivalence.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote equivalence report to {output_path}\n")
    print(f"Set match:   {set_match_count}/{len(questions)}")
    print(f"Order match: {order_match_count}/{len(questions)}")
    print(f"Tie flips: {tie_flip_count}  Real divergences: {real_divergence_count}")
    print(f"Max score diff: {max_score_diff_overall:.2e}")

    if set_match_count < len(questions) or real_divergence_count > 0:
        print("\nFAILED: set mismatches or real divergences found - see results/pgvector_equivalence.json")
        sys.exit(1)
    print("\nPASSED: pgvector retrieves the same chunks as FAISS for every question.")


if __name__ == "__main__":
    main()
