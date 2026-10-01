"""Uploads the existing 955-chunk corpus to GCS (one .txt object per chunk + a
manifest.jsonl) and imports it into a Discovery Engine (Vertex AI Search) data
store - the one-time provisioning step for the Vertex AI Search benchmark
comparison stretch goal.

Plain script (no typer), matching run_eval.py/measure_latency.py's convention.

Usage:
    python scripts/vertex_upload_corpus.py --config config/config.yaml
    python scripts/vertex_upload_corpus.py --skip-import   # upload to GCS only
    python scripts/vertex_upload_corpus.py --skip-upload   # import only (GCS objects already uploaded)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rag_receipts.config import load_config
from rag_receipts.retrieval.pipeline import load_metadata
from rag_receipts.vertex.upload import upload_corpus_to_gcs


def _import_documents(*, project_id: str, location: str, data_store_id: str, gcs_uri: str) -> str:
    """Imports the manifest into the Discovery Engine data store, polling the
    long-running operation to completion. Returns the operation name.

    Lazily imports google-cloud-discoveryengine - only needed when actually
    running the import, mirroring api/startup.py's lazy GCS client import.
    """
    from google.cloud import discoveryengine_v1 as discoveryengine

    client = discoveryengine.DocumentServiceClient()
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )
    request = discoveryengine.ImportDocumentsRequest(
        parent=parent,
        gcs_source=discoveryengine.GcsSource(
            input_uris=[gcs_uri],
            data_schema="document",
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
    )
    operation = client.import_documents(request=request)
    print(f"Import operation started: {operation.operation.name}")
    print("Waiting for import to complete...")
    operation.result()  # blocks until the long-running operation finishes
    return operation.operation.name


def _document_count(*, project_id: str, location: str, data_store_id: str) -> int:
    from google.cloud import discoveryengine_v1 as discoveryengine

    client = discoveryengine.DocumentServiceClient()
    parent = client.branch_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        branch="default_branch",
    )
    return sum(1 for _ in client.list_documents(parent=parent))


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    skip_upload = "--skip-upload" in sys.argv
    skip_import = "--skip-import" in sys.argv

    cfg = load_config(config_path)
    vertex_cfg = cfg.vertex
    if not vertex_cfg.gcs_bucket:
        raise SystemExit("vertex.gcs_bucket is not set in config.yaml - provision a bucket first")

    index_dir = Path(cfg.indexing.output.index_dir)
    metadata = load_metadata(index_dir / cfg.indexing.output.metadata_filename)
    print(f"Loaded {len(metadata)} chunks from index_metadata.parquet")

    manifest_uri = f"gs://{vertex_cfg.gcs_bucket}/{vertex_cfg.gcs_prefix}/manifest.jsonl"
    if not skip_upload:
        from google.cloud import storage

        client = storage.Client()
        manifest_uri = upload_corpus_to_gcs(
            metadata, client=client, bucket=vertex_cfg.gcs_bucket, prefix=vertex_cfg.gcs_prefix
        )
        print(f"Uploaded {len(metadata)} chunk objects + manifest to {manifest_uri}")
    else:
        print(f"Skipping upload; assuming manifest already exists at {manifest_uri}")

    import_operation_name = None
    document_count_after_import = None
    if not skip_import:
        if not vertex_cfg.project_id:
            raise SystemExit("vertex.project_id is not set in config.yaml - required for import")
        import_operation_name = _import_documents(
            project_id=vertex_cfg.project_id,
            location=vertex_cfg.location,
            data_store_id=vertex_cfg.data_store_id,
            gcs_uri=manifest_uri,
        )
        document_count_after_import = _document_count(
            project_id=vertex_cfg.project_id,
            location=vertex_cfg.location,
            data_store_id=vertex_cfg.data_store_id,
        )
        print(f"Data store now has {document_count_after_import} documents (expected {len(metadata)})")
    else:
        print("Skipping import (--skip-import)")

    stats = {
        "chunk_count": len(metadata),
        "bucket": vertex_cfg.gcs_bucket,
        "prefix": vertex_cfg.gcs_prefix,
        "manifest_uri": manifest_uri,
        "import_operation_name": import_operation_name,
        "document_count_after_import": document_count_after_import,
    }
    stats_path = Path("results/vertex_upload_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote upload stats to {stats_path}")


if __name__ == "__main__":
    main()
