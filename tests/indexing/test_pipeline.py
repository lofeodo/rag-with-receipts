import numpy as np
import pytest
from .helpers import FakeEncoder, make_chunks_df

from rag_receipts.config import AppConfig
from rag_receipts.indexing.config import EmbeddingConfig, IndexingConfig, IndexOutputConfig
from rag_receipts.indexing.embedder import Embedder
from rag_receipts.ingestion.config import CorpusConfig, IngestionConfig, LinkFollowSpec, QuestlineSpec
from rag_receipts.indexing.pipeline import build_embedding_texts, load_chunks, run_index


def _app_config(processed_dir: str) -> AppConfig:
    return AppConfig(
        corpus=CorpusConfig(
            source="osrs_wiki",
            api_base="https://example.invalid/api.php",
            license="CC BY-NC-SA 3.0",
            user_agent="test-agent",
        ),
        ingestion=IngestionConfig(
            categories=[],
            questline=QuestlineSpec(label="test", pages=[]),
            link_follow=LinkFollowSpec(),
            processed_dir=processed_dir,
        ),
        indexing=IndexingConfig(
            embedding=EmbeddingConfig(),
            output=IndexOutputConfig(),
        ),
    )


def test_build_embedding_texts_with_section_path():
    df = make_chunks_df(1)
    texts = build_embedding_texts(df)

    assert texts[0] == "Test page > Section A\n\nThis is synthetic chunk number 0."


def test_build_embedding_texts_without_section_path():
    df = make_chunks_df(2)
    texts = build_embedding_texts(df)

    assert texts[1] == "Test page\n\nThis is synthetic chunk number 1."


def test_run_index_end_to_end(tmp_path):
    df = make_chunks_df(5)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    df.to_parquet(processed_dir / "chunks.parquet", index=False)

    config = _app_config(str(processed_dir))
    fake = FakeEncoder(dim=6)
    embedder = Embedder(config=config.indexing.embedding, model=fake)

    index, metadata, stats = run_index(config, embedder=embedder)

    assert index.ntotal == 5
    assert list(metadata.columns) == list(df.columns)
    assert list(metadata["chunk_id"]) == list(df["chunk_id"])
    assert stats["chunk_count"] == 5
    assert stats["embedding_dim"] == 6
    assert stats["source_type_breakdown"] == {"prose": 5}


def test_run_index_row_alignment(tmp_path):
    df = make_chunks_df(3)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    df.to_parquet(processed_dir / "chunks.parquet", index=False)

    config = _app_config(str(processed_dir))
    fake = FakeEncoder(dim=6)
    embedder = Embedder(config=config.indexing.embedding, model=fake)

    index, metadata, _ = run_index(config, embedder=embedder)

    texts = build_embedding_texts(metadata)
    for i, text in enumerate(texts):
        expected = embedder.encode_passages([text])[0]
        actual = index.reconstruct(i)
        assert np.allclose(actual, expected)


def test_run_index_missing_file_raises(tmp_path):
    config = _app_config(str(tmp_path / "does_not_exist"))

    with pytest.raises(FileNotFoundError):
        run_index(config, embedder=Embedder(config=config.indexing.embedding, model=FakeEncoder()))


def test_run_index_empty_file_raises(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    empty = make_chunks_df(1).iloc[0:0]  # 0 rows, real Chunk schema columns
    empty.to_parquet(processed_dir / "chunks.parquet", index=False)
    config = _app_config(str(processed_dir))

    with pytest.raises(ValueError):
        run_index(config, embedder=Embedder(config=config.indexing.embedding, model=FakeEncoder()))


def test_load_chunks(tmp_path):
    df = make_chunks_df(2)
    path = tmp_path / "chunks.parquet"
    df.to_parquet(path, index=False)

    loaded = load_chunks(path)

    assert list(loaded["chunk_id"]) == list(df["chunk_id"])
