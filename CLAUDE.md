# CLAUDE.md — RAG With Receipts

Portfolio RAG project (1-week timeline) demonstrating retrieval accuracy, grounded/cited
answers, and latency-optimized inference — not basic chat-with-a-doc. Deployed on GCP
Cloud Run.

## Locked decisions

| Area | Decision |
|---|---|
| Corpus | OSRS Wiki (CC BY-NC-SA 3.0 — attribute in README, non-commercial use only). Scope: combat mechanics pages + 2 skill training guides with XP tables (Slayer, Herblore) + one bounded questline (requirements/rewards/walkthroughs) + the item/monster infobox pages those reference. ~200–300 pages. |
| Embeddings | Local `BAAI/bge-large-en-v1.5` via sentence-transformers, in-process. **Stretch:** benchmark against hosted Vertex `gemini-embedding-001` on the eval set; swap default only if it wins by a real margin. |
| Vector index | **FAISS** (flat index, file artifact) as the default — in-process, zero recurring cost, most technically substantive choice for the job-market signal the user wants. **Stretch:** add pgvector on Cloud SQL as a swappable second backend (config-driven) — strong "managed vector DB on GCP" resume line, done after the core pipeline works. |
| Reranker | Local cross-encoder, `BAAI/bge-reranker-base` baseline. Latency-optimization step compares a stronger (`bge-reranker-v2-m3`) and a faster (`ms-marco-MiniLM-L-6-v2`) variant on precision vs p95, and picks a point on that curve. |
| Generation + judge model | Claude Sonnet 5 for both, exposed as a config parameter (not hardcoded). **Stretch:** Haiku-vs-Sonnet generation comparison (latency/cost/correctness table), not a core metric. |
| Runtime shape | Embedding model, reranker, and FAISS index all run in-process inside the Cloud Run container. Only network hops in the hot path: the Claude generation call, and (only if the hosted-embedding stretch is adopted) the embedding call. |
| GCP deploy | Cloud Run (container, scales to zero) + Artifact Registry (image) + GCS (index artifacts, pulled on cold start) + Secret Manager (Anthropic key) + Cloud Logging (latency metrics). |
| Eval question mix | ~70% single-hop / extractive, ~30% multi-hop / synthesis (deliberately — single-hop questions don't discriminate between good and mediocre retrieval; multi-hop questions punish a pipeline that retrieves one relevant chunk and stops). User writes 30–50 Q/A pairs by hand. |

### Why not a fully-managed RAG service (Vertex AI RAG Engine / Vertex AI Search)?

The project's requirements — header-aware chunking, a chosen embedding model, a local/in-process
vector store, a reranking step, per-stage p50/p95 latency, a measured optimization, and a
grounding/entailment check on cited chunks — are exactly the internals a managed RAG service
hides. Using one would satisfy "a working assistant" but not the thing being demonstrated.
GCP is used instead as the **deployment substrate** (Cloud Run, GCS, Artifact Registry, Secret
Manager, Logging), while the pipeline itself is hand-built. See the stretch goal below for a
quantified comparison against the managed alternative.

## Step sequence

- [x] **Step 0 — Repo scaffold + CLAUDE.md** (this file, package skeleton, config, README stub)
- [ ] **Step 1 — Ingestion & chunking** ← in-depth plan below, ready to start
- [ ] Step 2 — Embedding & indexing (FAISS build, persisted artifacts)
- [ ] Step 3 — Retrieval pipeline (dense top-k + cross-encoder reranking)
- [ ] Step 4 — Generation (grounded answers with chunk citations)
- [ ] Step 5 — Eval harness (retrieval precision + LLM-as-judge correctness, results report)
- [ ] Step 6 — Grounding / hallucination check (entailment/overlap flag on cited chunks)
- [ ] Step 7 — Latency instrumentation + one measured optimization (reranker sweep)
- [ ] Step 8 — GCP deployment (Cloud Run, GCS, Artifact Registry, Secret Manager)
- [ ] Step 9 — README polish (architecture explanation, benchmark numbers front and center)
- [ ] Stretch — Vertex AI Search benchmark comparison (retrieval precision, correctness,
      grounding, end-to-end latency; note in the report that the retrieval/generation latency
      *split* is ours-only, Vertex only exposes end-to-end)
- [ ] Stretch — Haiku vs Sonnet generation comparison
- [ ] Stretch — pgvector-on-Cloud-SQL backend swap

## Workflow rules

- One step at a time. Don't plan future steps in detail up front.
- To start a step: user says so explicitly → enter plan mode → write a detailed, in-depth
  plan for that step *only* → append/update it under "Step detail" below → execute.
- Each step gets its own feature branch (`feat/<step-name>`), with small, continual commits —
  avoid large batched pushes.
- Update this file's checklist and "Step detail" section as work progresses, not just at
  step boundaries.
- Tell the user explicitly when a step is fully complete before starting the next one's
  in-depth plan.

## Step detail

### Step 1 — Ingestion & chunking (current)

See full plan in the conversation / PR description for this branch. Summary once implemented:
fetch a reproducible page list from the OSRS Wiki MediaWiki API, cache raw HTML, parse and
strip boilerplate (nav/TOC/edit links) while preserving infobox and table facts, chunk
header-aware with a token budget matched to `bge-large-en-v1.5` (512-token limit), emit
`chunks.parquet` + `chunks.jsonl` + `ingest_stats.json`.

_Status: not started. Awaiting user go-ahead to enter plan mode for this step._
