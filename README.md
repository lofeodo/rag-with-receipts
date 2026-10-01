# RAG With Receipts

A retrieval-augmented generation pipeline over a scoped slice of the [Old School RuneScape
Wiki](https://oldschool.runescape.wiki/), built to demonstrate retrieval accuracy, grounded
and cited answers, and latency-optimized inference — not just chat-with-a-doc.

**Status:** early scaffold. Architecture notes, setup instructions, and benchmark numbers
will land here as each pipeline stage is built — see [CLAUDE.md](CLAUDE.md) for the current
step and full plan.

## Why the OSRS Wiki

Chosen for a corpus that's deeply cross-referenced and numerically dense (exact XP values,
requirements, drop mechanics), which makes for a stronger eval set — both for multi-hop
retrieval questions and for the hallucination/grounding check. Content is CC BY-NC-SA 3.0;
used here for non-commercial, personal portfolio purposes.

## License

This project's code is MIT-licensed (see [LICENSE](LICENSE)). That covers the
pipeline, scripts, and config only — it does not relicense the OSRS Wiki
content itself, which remains CC BY-NC-SA 3.0, non-commercial, and attributed,
as described above.

## Architecture

Two offline stages build the index once; four online stages run per query;
`eval`/`telemetry` measure the pipeline and `api` serves it.

```mermaid
flowchart LR
    subgraph Offline["Offline (build-time)"]
        ING["Ingestion<br/>fetch + parse + chunk"]
        IDX["Indexing<br/>embed + FAISS build"]
        ING --> IDX
    end

    subgraph Online["Online (per query)"]
        Q["Query"] --> RET["Retrieval<br/>dense top-k"]
        RET --> RR["Rerank<br/>cross-encoder"]
        RR --> GEN["Generation<br/>Claude + citations"]
        GEN --> GRD["Grounding check<br/>NLI + lexical overlap"]
        GRD --> OUT["Answer + citations<br/>+ grounding verdicts"]
    end

    IDX -. "FAISS index +\nmetadata" .-> RET
```

| Stage | Key design choice | Package |
|---|---|---|
| Ingestion | MediaWiki API fetch over a scoped corpus (combat category + 2 skill-training guides + 1 questline + its one-hop-linked item/monster pages); header-aware chunking, ~380 target tokens | [`src/rag_receipts/ingestion/`](src/rag_receipts/ingestion/) |
| Indexing | Local `BAAI/bge-large-en-v1.5` embeddings; FAISS flat index, in-process, zero recurring cost | [`src/rag_receipts/indexing/`](src/rag_receipts/indexing/) |
| Retrieval | Dense top-30 → cross-encoder rerank to top-5; reranker is `ms-marco-MiniLM-L-6-v2`, adopted after a measured sweep (Pareto-dominates the original default) | [`src/rag_receipts/retrieval/`](src/rag_receipts/retrieval/) |
| Generation | Claude Sonnet 5, forced tool-use for structured per-claim citations, hallucinated-citation validation against the retrieved set | [`src/rag_receipts/generation/`](src/rag_receipts/generation/) |
| Grounding check | NLI cross-encoder entailment/contradiction + lexical overlap per citation, combined into a grounded/contradicted/ungrounded verdict | [`src/rag_receipts/grounding/`](src/rag_receipts/grounding/) |
| Eval harness *(offline)* | LLM-as-judge correctness + retrieval metrics + grounding, run over a 55-question hand-authored gold set | [`src/rag_receipts/eval/`](src/rag_receipts/eval/) |
| Telemetry *(offline)* | Per-stage p50/p95/mean latency instrumentation on the real hot path | [`src/rag_receipts/telemetry/`](src/rag_receipts/telemetry/) |
| API *(serving)* | FastAPI wrapper (`/query`, `/livez`, `/readyz`) deployed on Cloud Run — see [Deployment](#deployment) | [`src/rag_receipts/api/`](src/rag_receipts/api/) |

## Benchmarks

_Coming after the eval harness (Step 5) is built._

## Deployment

A live version runs on GCP Cloud Run:
**https://rag-receipts-api-374659103328.northamerica-northeast1.run.app**

The service is public but gated by a shared demo key (`/query` requires an
`X-Demo-Key` header matching a value kept in Secret Manager) — paste it into
the "Demo key" field on the page. Don't have one? Email
[daniel.lofeodo@gmail.com](mailto:daniel.lofeodo@gmail.com) to request it.
This was a
fallback from the original plan (Cloud Run's native IAP, gating access behind
Google sign-in): IAP's OAuth consent setup turns out to require the GCP
project to belong to an Organization, which a personal-account project isn't
part of, and even where that requirement is met the resulting brand only
admits users from the same Workspace domain — not the "any recruiter, any
Google account" goal — so a shared key is what's actually deployed.

### GCP resources used

| Resource | Role |
|---|---|
| Cloud Run | Serves the FastAPI app (`src/rag_receipts/api/`); scales to zero when idle, `max-instances=2` |
| Artifact Registry | Hosts the built container image |
| Cloud Build | Builds the image (embedding + reranker models baked in at build time) and pushes it |
| GCS | Holds `faiss.index` / `index_metadata.parquet`, pulled on cold start |
| Secret Manager | `anthropic-api-key`, `demo-api-key` |
| Cloud Logging | Captures stdout, including one structured JSON line per `/query` request with the full per-stage timing breakdown |

### Running locally

```
docker build -t rag-receipts-api .
docker run -p 8080:8080 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$(pwd)/data/index:/app/data/index:ro" \
  rag-receipts-api
```

Then open `http://localhost:8080/` (no `DEMO_API_KEY` set locally, so `/query`
is open) or hit `/livez`, `/readyz`, `/query` directly.

### Redeploying

```
gcloud builds submit --tag northamerica-northeast1-docker.pkg.dev/rag-with-receipts/rag-receipts/api:latest \
  --project rag-with-receipts

gcloud run deploy rag-receipts-api \
  --image=northamerica-northeast1-docker.pkg.dev/rag-with-receipts/rag-receipts/api:latest \
  --region=northamerica-northeast1 --project=rag-with-receipts \
  --allow-unauthenticated \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-api-key:latest,DEMO_API_KEY=demo-api-key:latest \
  --set-env-vars=INDEX_GCS_BUCKET=rag-with-receipts-index \
  --max-instances=2 --memory=4Gi --cpu=2 --timeout=60 --cpu-boost
```

### A note on latency: Cloud Run numbers are not the Step 7 numbers

Step 7's measured p50/p95 latency table was taken on local hardware with a
GPU (RTX 3060). Cloud Run has no GPU, so all inference runs on its allocated
2 vCPUs — and the gap is larger than "CPU vs GPU" alone would suggest. A few
real timed `/query` calls against the live deployment (steady-state, after
the first cold-start call):

| Stage | Local GPU (Step 7, `measure_latency.py`) | Local Docker, CPU | Cloud Run, 2 vCPU |
|---|---|---|---|
| embed_query | 50ms p50 | ~100ms | ~500ms |
| rerank | 111ms p50 (MiniLM) | ~700ms | ~5000ms |
| generate | 2737ms p50 | ~3400ms | ~3400ms |
| total | 3510ms p50 | ~4200ms | ~9000ms |

Rerank is the stage that degrades the most on Cloud Run — about 7x slower
than the same model on the same machine's CPU outside a container, and ~45x
slower than the GPU numbers Step 7 reported. This wasn't chased further in
Step 8 (which is a deployment step, not a second latency-optimization pass) -
documented here as a known, measured gap rather than papered over. Candidates
for a future pass, not done here: tuning `OMP_NUM_THREADS`/torch thread
count for the 2-vCPU environment, or bumping Cloud Run's CPU allocation.
