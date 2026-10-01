"""Builds the per-chunk GCS upload content + Discovery Engine import manifest
for the Vertex AI Search comparison stretch goal.

One GCS object per existing chunk (955 objects, not per raw page) - a locked
decision (see CLAUDE.md's Vertex stretch-goal status note): this keeps Vertex's
search results scorable against the existing gold_chunk_ids via exact chunk_id
match, at the cost of not exercising Vertex's own chunking.
"""

from __future__ import annotations

import pandas as pd

from rag_receipts.ingestion.utils import breadcrumb


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
