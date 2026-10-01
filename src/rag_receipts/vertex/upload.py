"""Builds the per-chunk GCS upload content + Discovery Engine import manifest
for the Vertex AI Search comparison stretch goal.

One GCS object per existing chunk (955 objects, not per raw page) - a locked
decision (see CLAUDE.md's Vertex stretch-goal status note): this keeps Vertex's
search results scorable against the existing gold_chunk_ids via exact chunk_id
match, at the cost of not exercising Vertex's own chunking.
"""

from __future__ import annotations

import json
from typing import Protocol

import pandas as pd

from rag_receipts.ingestion.utils import breadcrumb


class UploadBlobLike(Protocol):
    def upload_from_string(self, data: str, content_type: str = "text/plain") -> None: ...


class UploadBucketLike(Protocol):
    def blob(self, blob_name: str) -> UploadBlobLike: ...


class GCSUploadClientLike(Protocol):
    def bucket(self, bucket_name: str) -> UploadBucketLike: ...


def chunk_text_for_upload(page_title: str, section_path: str, text: str) -> str:
    """Same text form already embedded (Step 2) and reranked (Step 3) on, so
    Vertex sees identical content to our own pipeline."""
    return f"{breadcrumb(page_title, section_path)}\n\n{text}"


def build_manifest_records(metadata: pd.DataFrame, *, bucket: str, prefix: str) -> list[dict]:
    """One Discovery Engine unstructured-document-with-metadata import record per chunk.

    chunk_id is set as the literal Document id *and* duplicated into
    structData.chunk_id - redundant on purpose, so chunk_id resolution has two
    independent channels to cross-check (same "don't trust a single channel of
    untrusted provenance" discipline as Step 4's citation resolution).
    """
    records = []
    for row in metadata.itertuples():
        chunk_id = row.chunk_id
        records.append(
            {
                "id": chunk_id,
                "structData": {
                    "chunk_id": chunk_id,
                    "page_title": row.page_title,
                    "url": row.url,
                    "section_path": row.section_path,
                    "heading_anchor": row.heading_anchor,
                    "source_type": row.source_type,
                    "token_count": int(row.token_count),
                },
                "content": {
                    "mimeType": "text/plain",
                    "uri": f"gs://{bucket}/{prefix}/chunks/{chunk_id}.txt",
                },
            }
        )
    return records


def upload_corpus_to_gcs(
    metadata: pd.DataFrame,
    *,
    client: GCSUploadClientLike,
    bucket: str,
    prefix: str,
) -> str:
    """Uploads one chunks/<chunk_id>.txt object per chunk plus manifest.jsonl.

    Returns the manifest's gs:// URI (what Discovery Engine's import_documents
    call is pointed at).
    """
    gcs_bucket = client.bucket(bucket)

    for row in metadata.itertuples():
        text = chunk_text_for_upload(row.page_title, row.section_path, row.text)
        blob = gcs_bucket.blob(f"{prefix}/chunks/{row.chunk_id}.txt")
        blob.upload_from_string(text, content_type="text/plain")

    records = build_manifest_records(metadata, bucket=bucket, prefix=prefix)
    manifest_text = "\n".join(json.dumps(record) for record in records)
    manifest_blob_name = f"{prefix}/manifest.jsonl"
    gcs_bucket.blob(manifest_blob_name).upload_from_string(
        manifest_text, content_type="application/jsonl"
    )

    return f"gs://{bucket}/{manifest_blob_name}"
