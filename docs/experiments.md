# Experiments

Three measured side-experiments on top of the core pipeline. Each used the same 55-question gold set. Raw data is in [`results/`](../results/).

- [Vertex AI Search vs hand-built](#vertex-ai-search-vs-hand-built)
- [Haiku 4.5 vs Sonnet 5](#haiku-45-vs-sonnet-5)
- [pgvector on Cloud SQL vs FAISS](#pgvector-on-cloud-sql-vs-faiss)

## Vertex AI Search vs hand-built

This project hand-builds retrieval and generation on the argument that chunking, embedding choice, a local vector store, reranking, per-stage latency and a grounding check are exactly the internals a managed service hides (see [architecture.md](architecture.md#why-not-a-managed-rag-service)). This experiment backs that with numbers. Vertex AI Search (Discovery Engine) was provisioned over the same 955-chunk corpus and measured two ways:

- **Config A (retrieval only):** Vertex's Search API, then our own Generator / Judge / GroundingChecker.
- **Config B (fully managed):** Vertex's Answer API end to end, also run through our GroundingChecker, alongside Vertex's self-reported `grounding_score`.

| Metric | Baseline (ours) | Config A (Vertex Search) | Config B (Vertex Answer) |
|---|---|---|---|
| Retrieval hit rate | 0.96 | **0.98** | 0.96 |
| Retrieval recall | 0.88 | **0.96** | 0.94 |
| Retrieval MRR | 0.752 | 0.823 | **0.858** |
| Correctness accuracy | 0.836 | **0.927** | 0.855 |
| Grounded rate (our checker) | 0.948 | **0.968** | 0.838 |
| Vertex's own grounding_score | n/a | n/a | 0.688 |
| End-to-end latency (p50 / p95) | 3.26s / 7.58s | 5.02s / 9.31s | **6.39s / 8.88s** |

Vertex's managed retrieval scored better than the hand-built pipeline on every retrieval and correctness metric. That is a genuine result, not cherry-picked. It is also slower end to end, and the asymmetries below limit how far it generalizes.

**Four structural asymmetries** (also recorded in `results/vertex_comparison.json`):

1. **Chunking wasn't compared.** Vertex ingested our already-chunked 955 objects, so only its retrieval and ranking were measured, not its own chunking.
2. **Latency isn't apples to apples.** The baseline exposes per-stage latency. Vertex exposes one end-to-end number per call.
3. **Config B's retrieval numbers mean something narrower.** They're scored against the chunk_ids Vertex actually cited (deduplicated; a chunk is cited once per claim, which inflated recall above 1.0 before it was caught and fixed).
4. **Config B has no clean abstention signal.** Vertex writes a prose "no information found" answer as normal text. All 5 unanswerable questions scored `incorrectly_answered` for config B. Had it abstained correctly, its accuracy would be 0.945, so its correctness gap is an artifact of the missing signal.

Also found while provisioning: an undocumented 10 LLM-requests/minute quota on the Answer API (the client now paces and retries), and `struct_data` on a live response is a proto-plus `MapComposite`, not the protobuf `Struct` the docs implied. Query cost was $0 (inside the 10,000 free queries/month). Everything was torn down immediately afterwards.

## Haiku 4.5 vs Sonnet 5

Only the generation model changes. Retrieval, the judge (always Sonnet 5), and the grounding checker are held fixed.

| Metric | Claude Sonnet 5 | Claude Haiku 4.5 |
|---|---|---|
| Correctness accuracy | **0.836** | 0.818 |
| Single-hop (n=35) | **1.000** | 0.971 |
| Multi-hop (n=15) | **0.467** | 0.400 |
| Grounded rate | 0.951 | **0.973** |
| Latency p50 / p95 | 2.55s / 6.17s | **2.38s / 3.90s** |
| Generation + judge cost (55 questions) | $0.683 | **$0.348** |

Retrieval metrics are identical across both arms, which confirms only the generation model varied.

**Not a clean win for either.** Sonnet's accuracy edge (about one question in 55) is inside the run-to-run non-determinism band. Haiku is better grounded, faster at the tail, and about half the cost. They also fail differently: Sonnet produced 1 answer where it should have abstained; Haiku never did but abstained more often when it could have answered (5 vs 3). Sonnet 5 stays the default since one run is not enough to change a locked decision.

**Fairness is logged, not claimed.** Each arm's `request_config` (model, `max_tokens`, system prompt, tool schema, forced `tool_choice`) is written to `results/generation_model_comparison.json`. Two asymmetries remain even with an identical request: neither call sets `thinking`, but Sonnet 5 thinks adaptively by default and Haiku 4.5 doesn't think; and neither pins `temperature`, because Sonnet 5 would reject it while Haiku would accept it. Pricing ($2/$10 vs $1/$5 per 1M input/output tokens) is Anthropic's first-party rate captured 2026-10-02.

## pgvector on Cloud SQL vs FAISS

The vector store is a config-driven, swappable backend (`indexing.vector_index: faiss | pgvector`) behind a small `VectorStore` Protocol. `Retriever.from_config` is the single place that picks one.

**This is deliberately not a quality comparison.** pgvector runs exact nearest-neighbor search (no HNSW/IVFFlat) over the same embeddings, which is guaranteed to retrieve the same chunks. The live run confirmed it:

| Metric | Value |
|---|---|
| Set match (chunk_ids) | 55/55 |
| Order match | 55/55 |
| Max score diff (FAISS vs pgvector) | 1.71e-07 |

What does differ is latency, a network round trip to Cloud SQL vs an in-process lookup, measured with identical code:

| Stage | FAISS p50 | pgvector p50 |
|---|---|---|
| dense_search | 1.3ms | 98.2ms |
| retrieval_total | 333ms | 471ms |
| end_to_end | 2884ms | 3010ms |

Dense search is about 75x slower over the network, but end to end grows only about 4% because generation dominates. The Cloud SQL instance was provisioned, measured and torn down. The live deployment stays on FAISS.

Design notes:

- **Auth is IAM database authentication**, so no password secret was ever created.
- **Driver:** Cloud SQL Python Connector + `pg8000`. `pgvector`'s `register_vector` is incompatible with the DB-API connection the Connector returns, so vectors are passed as text literals cast to `::vector`.
- **Setup gotchas:** new Cloud SQL instances default to Enterprise Plus, which rejects `db-f1-micro` (use `--edition=enterprise`). `CREATE EXTENSION vector` needs superuser, and Postgres 15+ also needs an explicit `GRANT ALL ON SCHEMA public`. Both are one-time setup steps.
- **Guarded build:** `indexing.pgvector.enabled_for_build` defaults to `false` so a stale config can't write to a torn-down instance. The live integration test asserts via `EXPLAIN` that only a sequential scan is used.
