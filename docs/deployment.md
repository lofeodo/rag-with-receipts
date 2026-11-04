# Past deployment (GCP Cloud Run, retired)

From October to November 2026 the app ran on GCP Cloud Run as a public demo. It has been shut down and the GCP project deleted to stop hosting costs; the project now runs locally only (see [running-locally.md](running-locally.md)). This page records how it was deployed and what the deployment taught.

**Access model.** `/query` was gated by a shared demo key (`X-Demo-Key` header, checked against the `DEMO_API_KEY` env var) plus a per-IP rate limiter, to bound API cost. The original plan was Cloud Run's native IAP, open to any Google account, but `gcloud iap oauth-brands create` failed with "Project must belong to an organization", and personal GCP projects have none. Even with an org, that brand is internal-only. The key check is still in the code and stays off unless `DEMO_API_KEY` is set.

## GCP resources (all deleted)

| Resource | Role |
|---|---|
| Cloud Run | Served the FastAPI app (`src/rag_receipts/api/`); scaled to zero, `max-instances=2` |
| Artifact Registry | Hosted the container image |
| Cloud Build | Built the image (embedding and reranker models baked in at build time) |
| GCS | Held `faiss.index` and `index_metadata.parquet`, pulled on cold start (the index is now committed under `data/index/`) |
| Secret Manager | `anthropic-api-key`, `demo-api-key` |
| Cloud Logging | Captured stdout, including one structured JSON line per `/query` with full per-stage timing |

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
