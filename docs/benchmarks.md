# Benchmarks

All numbers come from one clean run of the full 55-question hand-authored gold set (`data/eval/qa_pairs.json`) under the current default config (`ms-marco-MiniLM-L-6-v2` reranker). Raw data lives in [`results/`](../results/). Reproduce with `python scripts/run_eval.py` and `python scripts/measure_latency.py`.

## Retrieval

| | Overall | Single-hop (n=35) | Multi-hop (n=15) |
|---|---|---|---|
| Hit rate | 0.96 | 0.97 | 0.93 |
| Recall | 0.88 | 0.97 | 0.67 |
| MRR | 0.75 | 0.81 | 0.61 |

Precision is deliberately low by construction: the denominator is `top_k_final=5` against a 1-2 chunk gold set, so even perfect retrieval can't clear about 0.2-0.4.

## Correctness (LLM-as-judge)

| | Overall | Single-hop | Multi-hop |
|---|---|---|---|
| Accuracy | 0.84 | 1.00 | 0.47 |

The multi-hop number is n=15, small enough that a couple of borderline verdicts move it several points. Treat it as "meaningfully worse than single-hop", not a precise figure. The gap is mostly a **retrieval** story: most non-correct multi-hop cases trace back to the top-k missing one of two gold chunks, and generation correctly declined rather than fabricating the missing fact.

Generation and judging call the live Claude API without `temperature` pinned (Sonnet 5's adaptive thinking rejects sampling params), so re-runs shift a few borderline verdicts. Retrieval is deterministic local inference and reproduces exactly.

Label counts: `correct` 42, `partially_correct` 1, `incorrect` 3, `incorrectly_abstained` 4, `correct_abstention` 4, `incorrectly_answered` 1. The one `incorrectly_answered` is a gold-labelling nuance, not a hallucination: the question (full Recipe for Disaster rewards) is "unanswerable" in the gold set, but the corpus holds partial reward figures and the model said so explicitly.

## Grounding (citation-level NLI + overlap)

| Grounded | Contradicted | Ungrounded | Fully-grounded answers |
|---|---|---|---|
| 0.95 | 0.05 | 0.00 | 0.91 |

77 citations checked across 44 answers. Read `contradicted` as "flagged for human review". Manual inspection found the flagged cases were false positives: the general-domain NLI model misreads this corpus's flattened-table, telegraphic chunk style. Thresholds were deliberately not tuned against a few anecdotes.

## Reranker sweep (what justified the current default)

| Model | Recall | Hit rate | MRR | Rerank p50 | Rerank p95 |
|---|---|---|---|---|---|
| `bge-reranker-base` (original default) | 0.83 | 0.90 | 0.70 | 552ms | 567ms |
| `bge-reranker-v2-m3` (stronger) | 0.90 | 0.98 | 0.84 | 2775ms | 3331ms |
| **`ms-marco-MiniLM-L-6-v2` (adopted)** | **0.88** | **0.96** | **0.75** | **111ms** | **128ms** |

The adopted model beats the original default on every retrieval metric and is about 5x faster, so it wasn't a tradeoff call. `bge-reranker-v2-m3` is more accurate still, but about 26x slower than the adopted default, which isn't justified when end-to-end latency is already dominated by generation. This is a measured result on this corpus (many terse, tabular chunks, n=50 scored questions), not a general claim about MiniLM vs BGE rerankers.

## Latency (per stage, local GPU)

| Stage | p50 | p95 |
|---|---|---|
| embed_query | 80ms | 154ms |
| dense_search | 0.3ms | 0.7ms |
| rerank | 276ms | 340ms |
| retrieval_total | 349ms | 425ms |
| generate | 2918ms | 7224ms |
| end_to_end | 3256ms | 7577ms |

Generation (the Claude API round trip) dominates end-to-end latency, not retrieval. Treat p95 as directional: it's roughly the 52nd-highest of 55 samples.

Methodology lessons recorded along the way: an early latency run overlapped with the reranker sweep on the same GPU and showed a 2.7x inflated rerank time, caught by comparing against the sweep's isolated measurement of the same model and re-run alone. Never run two GPU-bound benchmarks concurrently.

For Cloud Run numbers see [deployment.md](deployment.md#latency-cloud-run-vs-local-hardware). For stretch comparisons see [experiments.md](experiments.md).
