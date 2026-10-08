# Deployment (GCP Cloud Run)

Live: **https://rag-receipts-api-374659103328.northamerica-northeast1.run.app**

The service is public but gated by a shared demo key (`/query` requires an `X-Demo-Key` header). Paste it into the "Demo key" field on the page. To request one, email [daniel.lofeodo@gmail.com](mailto:daniel.lofeodo@gmail.com).

**Why a shared key and not IAP?** The original plan was Cloud Run's native IAP, open to any Google account. `gcloud iap oauth-brands create` failed with "Project must belong to an organization", and personal GCP projects have none. Even with an org, that brand is internal-only. A shared key plus a per-IP rate limiter was the fallback.

## GCP resources

| Resource | Role |
|---|---|
| Cloud Run | Serves the FastAPI app (`src/rag_receipts/api/`); scales to zero, `max-instances=2` |
| Artifact Registry | Hosts the container image |
| Cloud Build | Builds the image (embedding and reranker models baked in at build time) |
| GCS | Holds `faiss.index` and `index_metadata.parquet`, pulled on cold start |
| Secret Manager | `anthropic-api-key`, `demo-api-key` |
| Cloud Logging | Captures stdout, including one structured JSON line per `/query` with full per-stage timing |

## Redeploying

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

## Latency: Cloud Run vs local hardware

The latency numbers in [benchmarks.md](benchmarks.md) were measured on a local RTX 3060. Cloud Run has no GPU, so inference runs on 2 vCPUs. The gap is larger than CPU-vs-GPU alone would explain. Steady-state calls, after the cold-start call:

| Stage | Local GPU | Local Docker, CPU | Cloud Run, 2 vCPU |
|---|---|---|---|
| embed_query | 80ms p50 | ~100ms | ~500ms |
| rerank | 276ms p50 (MiniLM) | ~700ms | ~5000ms |
| generate | 2918ms p50 | ~3400ms | ~3400ms |
| total | 3256ms p50 | ~4200ms | ~9000ms |

Rerank degrades most on Cloud Run: about 7x slower than the same model on the same machine's CPU outside a container, and about 18x slower than local GPU. Not chased further. Candidates: tuning `OMP_NUM_THREADS` / torch thread count for 2 vCPUs, or more CPU.

## Gotchas found while deploying

- `/healthz` returns a Google-branded 404 on `*.run.app`: Google's frontend intercepts that exact path before it reaches the container. The liveness route is `/livez`.
- `pandas` / `pyarrow` were declared only under the `ingest` extra but are needed at retrieval time, so the container crashed on startup until they moved to the `index` extra. Caught by running the built image locally before any GCP resource existed.
