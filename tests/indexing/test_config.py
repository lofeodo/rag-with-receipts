from rag_receipts.config import load_config
from rag_receipts.indexing.config import _build_indexing_config


def test_defaults_when_section_empty():
    cfg = _build_indexing_config({})

    assert cfg.vector_index == "faiss"
    assert cfg.embedding.model_name == "BAAI/bge-large-en-v1.5"
    assert cfg.embedding.batch_size == 32
    assert cfg.embedding.normalize_embeddings is True
    assert cfg.embedding.device == "auto"
    assert cfg.embedding.query_instruction == (
        "Represent this sentence for searching relevant passages: "
    )
    assert cfg.output.index_dir == "data/index"
    assert cfg.output.index_filename == "faiss.index"
    assert cfg.output.metadata_filename == "index_metadata.parquet"
    assert cfg.output.stats_path == "data/processed/index_stats.json"


def test_overrides_are_applied():
    raw = {
        "embedding_model": "some/other-model",
        "batch_size": 64,
        "device": "cpu",
        "output": {"index_dir": "custom/index"},
    }

    cfg = _build_indexing_config(raw)

    assert cfg.embedding.model_name == "some/other-model"
    assert cfg.embedding.batch_size == 64
    assert cfg.embedding.device == "cpu"
    assert cfg.output.index_dir == "custom/index"
    # untouched nested keys keep their defaults
    assert cfg.output.index_filename == "faiss.index"


def test_load_config_populates_indexing_section():
    app_config = load_config("config/config.yaml")

    assert app_config.indexing.embedding.model_name == "BAAI/bge-large-en-v1.5"
    assert app_config.indexing.vector_index == "faiss"
