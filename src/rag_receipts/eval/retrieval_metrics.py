"""Retrieval precision/recall/MRR/hit metrics against a gold chunk_id set (Step 5).

These measure the pipeline's actual final output (Retriever.retrieve(), i.e. post-rerank
top_k_final) - the same chunks generation actually receives. Comparing dense-only vs
reranked precision is deliberately out of scope here; that's Step 7's reranker sweep.
"""

from __future__ import annotations

from rag_receipts.eval.models import RetrievalScore
from rag_receipts.retrieval.models import RetrievedChunk


def precision_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in gold_ids)
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in gold_ids)
    return hits / len(gold_ids)


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def hit(retrieved_ids: list[str], gold_ids: set[str]) -> bool:
    return any(cid in gold_ids for cid in retrieved_ids)


def score_retrieval(retrieved: list[RetrievedChunk], gold_chunk_ids: list[str]) -> RetrievalScore:
    retrieved_ids = [c.chunk_id for c in retrieved]
    gold_ids = set(gold_chunk_ids)
    return RetrievalScore(
        precision=precision_at_k(retrieved_ids, gold_ids),
        recall=recall_at_k(retrieved_ids, gold_ids),
        mrr=mrr(retrieved_ids, gold_ids),
        hit=hit(retrieved_ids, gold_ids),
        retrieved_chunk_ids=retrieved_ids,
    )
