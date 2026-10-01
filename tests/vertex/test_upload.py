import json

import pandas as pd

from rag_receipts.vertex.upload import (
    build_manifest_records,
    chunk_text_for_upload,
    upload_corpus_to_gcs,
)


class FakeBlob:
    def __init__(self, name: str, store: dict):
        self.name = name
        self._store = store

    def upload_from_string(self, data: str, content_type: str = "text/plain") -> None:
        self._store[self.name] = (data, content_type)


class FakeBucket:
    def __init__(self, store: dict):
        self._store = store

    def blob(self, blob_name: str) -> FakeBlob:
        return FakeBlob(blob_name, self._store)


class FakeGCSClient:
    def __init__(self):
        self.uploads: dict[str, tuple[str, str]] = {}

    def bucket(self, bucket_name: str) -> FakeBucket:
        return FakeBucket(self.uploads)


def make_metadata_df(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": f"Attack_range__{i:03d}",
                "page_title": "Attack range",
                "url": "https://oldschool.runescape.wiki/w/Attack_range",
                "section_path": "Overview",
                "heading_anchor": "Overview",
                "token_count": 100 + i,
                "source_type": "prose",
                "text": f"chunk text {i}",
            }
            for i in range(n)
        ]
    )


def test_chunk_text_for_upload_matches_breadcrumb_form():
    text = chunk_text_for_upload("Attack range", "Overview", "Some body text.")

    assert text == "Attack range > Overview\n\nSome body text."


def test_chunk_text_for_upload_no_section_path():
    text = chunk_text_for_upload("Attack range", "", "Some body text.")

    assert text == "Attack range\n\nSome body text."


def test_build_manifest_records_one_per_row():
    metadata = make_metadata_df(3)

    records = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert len(records) == 3


def test_build_manifest_records_fields():
    metadata = make_metadata_df(1)

    [record] = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert record["id"] == "Attack_range__000"
    assert record["structData"]["chunk_id"] == "Attack_range__000"
    assert record["structData"]["page_title"] == "Attack range"
    assert record["structData"]["url"] == "https://oldschool.runescape.wiki/w/Attack_range"
    assert record["structData"]["section_path"] == "Overview"
    assert record["structData"]["heading_anchor"] == "Overview"
    assert record["structData"]["source_type"] == "prose"
    assert record["structData"]["token_count"] == 100
    assert record["content"] == {
        "mimeType": "text/plain",
        "uri": "gs://my-bucket/corpus/chunks/Attack_range__000.txt",
    }


def test_build_manifest_records_chunk_id_survives_as_both_id_and_struct_data():
    metadata = make_metadata_df(1)

    [record] = build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")

    assert record["id"] == record["structData"]["chunk_id"]


def test_upload_corpus_to_gcs_uploads_one_object_per_chunk_plus_manifest():
    metadata = make_metadata_df(3)
    client = FakeGCSClient()

    manifest_uri = upload_corpus_to_gcs(metadata, client=client, bucket="my-bucket", prefix="corpus")

    assert manifest_uri == "gs://my-bucket/corpus/manifest.jsonl"
    assert len(client.uploads) == 4  # 3 chunk .txt objects + 1 manifest.jsonl
    for i in range(3):
        assert f"corpus/chunks/Attack_range__{i:03d}.txt" in client.uploads


def test_upload_corpus_to_gcs_chunk_text_matches_chunk_text_for_upload():
    metadata = make_metadata_df(1)
    client = FakeGCSClient()

    upload_corpus_to_gcs(metadata, client=client, bucket="my-bucket", prefix="corpus")

    uploaded_text, content_type = client.uploads["corpus/chunks/Attack_range__000.txt"]
    assert uploaded_text == chunk_text_for_upload("Attack range", "Overview", "chunk text 0")
    assert content_type == "text/plain"


def test_upload_corpus_to_gcs_manifest_is_valid_jsonl_matching_build_manifest_records():
    metadata = make_metadata_df(2)
    client = FakeGCSClient()

    upload_corpus_to_gcs(metadata, client=client, bucket="my-bucket", prefix="corpus")

    manifest_text, content_type = client.uploads["corpus/manifest.jsonl"]
    assert content_type == "application/jsonl"
    lines = [json.loads(line) for line in manifest_text.splitlines()]
    assert lines == build_manifest_records(metadata, bucket="my-bucket", prefix="corpus")
