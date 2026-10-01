from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_receipts.api.startup import ensure_index_artifacts


def _fake_config(index_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        indexing=SimpleNamespace(
            output=SimpleNamespace(
                index_dir=str(index_dir),
                index_filename="faiss.index",
                metadata_filename="index_metadata.parquet",
            )
        )
    )


class FakeBlob:
    def __init__(self, name: str, *, fail: bool = False):
        self.name = name
        self.fail = fail
        self.downloaded_to: str | None = None

    def download_to_filename(self, filename: str) -> None:
        if self.fail:
            raise OSError("simulated GCS failure")
        self.downloaded_to = filename
        Path(filename).write_bytes(b"fake artifact bytes")


class FakeBucket:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.blobs_requested: list[str] = []

    def blob(self, blob_name: str) -> FakeBlob:
        self.blobs_requested.append(blob_name)
        return FakeBlob(blob_name, fail=self.fail)


class FakeGCSClient:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.buckets_requested: list[str] = []

    def bucket(self, bucket_name: str) -> FakeBucket:
        self.buckets_requested.append(bucket_name)
        return FakeBucket(fail=self.fail)


class ExplodingGCSClient:
    """Used to prove ensure_index_artifacts never touches GCS in the no-op cases."""

    def bucket(self, bucket_name: str):
        raise AssertionError("GCS client should not have been called")


def test_noop_when_bucket_env_var_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("INDEX_GCS_BUCKET", raising=False)
    config = _fake_config(tmp_path)

    ensure_index_artifacts(config, client=ExplodingGCSClient())

    assert list(tmp_path.iterdir()) == []


def test_noop_when_files_already_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_GCS_BUCKET", "rag-with-receipts-index")
    (tmp_path / "faiss.index").write_bytes(b"existing index")
    (tmp_path / "index_metadata.parquet").write_bytes(b"existing metadata")
    config = _fake_config(tmp_path)

    ensure_index_artifacts(config, client=ExplodingGCSClient())

    assert (tmp_path / "faiss.index").read_bytes() == b"existing index"


def test_downloads_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_GCS_BUCKET", "rag-with-receipts-index")
    config = _fake_config(tmp_path)
    client = FakeGCSClient()

    ensure_index_artifacts(config, client=client, prefix="index/")

    assert client.buckets_requested == ["rag-with-receipts-index"]
    assert (tmp_path / "faiss.index").exists()
    assert (tmp_path / "index_metadata.parquet").exists()


def test_downloads_only_the_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_GCS_BUCKET", "rag-with-receipts-index")
    (tmp_path / "faiss.index").write_bytes(b"already here")
    config = _fake_config(tmp_path)
    client = FakeGCSClient()

    ensure_index_artifacts(config, client=client)

    assert (tmp_path / "faiss.index").read_bytes() == b"already here"
    assert (tmp_path / "index_metadata.parquet").exists()


def test_raises_clear_error_on_download_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_GCS_BUCKET", "rag-with-receipts-index")
    config = _fake_config(tmp_path)
    client = FakeGCSClient(fail=True)

    with pytest.raises(RuntimeError, match="rag-with-receipts-index"):
        ensure_index_artifacts(config, client=client)
