"""Cold-start index-artifact pull from GCS (Step 8).

Mirrors the locked deploy decision in CLAUDE.md: FAISS index + metadata are
built by scripts/build_index.py and live in GCS as file artifacts, pulled
into the Cloud Run container on cold start rather than baked into the image
(the image is reusable across corpus rebuilds; only the bucket needs
re-uploading when the index changes).

ensure_index_artifacts() is a no-op whenever INDEX_GCS_BUCKET is unset or the
local files already exist, so every existing script/test that constructs
Retriever.from_config(cfg) directly against local data/index/ files keeps
working unchanged - this is purely additive, local-dev/test behavior is
unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from rag_receipts.config import AppConfig


class BlobLike(Protocol):
    def download_to_filename(self, filename: str) -> None: ...


class GCSClientLike(Protocol):
    def bucket(self, bucket_name: str) -> "BucketLike": ...


class BucketLike(Protocol):
    def blob(self, blob_name: str) -> BlobLike: ...


def _default_client() -> GCSClientLike:
    from google.cloud import storage  # imported lazily - only needed when actually pulling

    return storage.Client()


def ensure_index_artifacts(
    config: AppConfig,
    *,
    client: GCSClientLike | None = None,
    bucket_env_var: str = "INDEX_GCS_BUCKET",
    prefix: str = "index/",
) -> None:
    bucket_name = os.environ.get(bucket_env_var)
    if not bucket_name:
        return  # local dev / existing scripts: use whatever is already on disk

    index_dir = Path(config.indexing.output.index_dir)
    index_path = index_dir / config.indexing.output.index_filename
    metadata_path = index_dir / config.indexing.output.metadata_filename

    missing = [p for p in (index_path, metadata_path) if not p.exists()]
    if not missing:
        return

    gcs = client or _default_client()
    bucket = gcs.bucket(bucket_name)
    index_dir.mkdir(parents=True, exist_ok=True)

    for local_path in missing:
        blob_name = f"{prefix}{local_path.name}"
        blob = bucket.blob(blob_name)
        try:
            blob.download_to_filename(str(local_path))
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable context below
            raise RuntimeError(
                f"Failed to download gs://{bucket_name}/{blob_name} to {local_path}: {exc}"
            ) from exc
