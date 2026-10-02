# RAG With Receipts - Cloud Run image (Step 8).
#
# Cloud Run has no GPU, so torch is installed explicitly from the CPU-only wheel
# index first (same lesson as CLAUDE.md's Step 2 status note, applied the other
# direction: there the fix was getting a CUDA wheel on a GPU box; here there is
# no GPU at all, so CPU-only is the correct install, not a fallback).
#
# The embedding and reranker models are baked into the image at build time
# (see the HF_HOME cache step below) rather than downloaded at container
# startup - this keeps the locked runtime-shape decision true even on a brand
# new container: the only network calls in the hot path are GCS (index pull)
# and the Anthropic API, never Hugging Face Hub.
#
# results/ is baked in too (small, committed JSON) so GET /benchmarks has
# something to read in the deployed container, not just in local dev.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src/ ./src/
COPY config/ ./config/
COPY static/ ./static/
COPY results/ ./results/

RUN pip install -e ".[index,serve,gcp]"

# Bake the embedding + reranker models (read straight from config.yaml, so the
# image always matches whatever models are actually configured - no separate
# hardcoded model list to drift out of sync, as happened when Step 7 changed
# the default reranker).
RUN python -c "\
import yaml; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
cfg = yaml.safe_load(open('config/config.yaml')); \
SentenceTransformer(cfg['indexing']['embedding_model']); \
CrossEncoder(cfg['retrieval']['reranker_model']); \
print('models cached')"

# Set only now (not during the bake step above, which needs network to
# actually populate the cache) - forces every runtime model load to use the
# baked cache with zero Hugging Face Hub calls, not just a cache hit after a
# version-check network round trip.
ENV HF_HUB_OFFLINE=1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn rag_receipts.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
