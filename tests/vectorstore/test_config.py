from rag_receipts.config import load_config
from rag_receipts.vectorstore.config import _build_pgvector_config


def test_defaults_when_section_empty():
    cfg = _build_pgvector_config({})

    assert cfg.enabled_for_build is False
    assert cfg.project_id == ""
    assert cfg.region == "northamerica-northeast1"
    assert cfg.instance_name == "rag-receipts-pgvector"
    assert cfg.database == "rag_receipts"
    assert cfg.table_name == "chunks"
    assert cfg.iam_user == ""
    assert cfg.embedding_dim == 1024


def test_overrides_are_applied():
    raw = {
        "enabled_for_build": True,
        "project_id": "some-project",
        "region": "us-central1",
        "instance_name": "custom-instance",
        "database": "custom_db",
        "table_name": "custom_chunks",
        "iam_user": "someone@example.com",
        "embedding_dim": 768,
    }

    cfg = _build_pgvector_config(raw)

    assert cfg.enabled_for_build is True
    assert cfg.project_id == "some-project"
    assert cfg.region == "us-central1"
    assert cfg.instance_name == "custom-instance"
    assert cfg.database == "custom_db"
    assert cfg.table_name == "custom_chunks"
    assert cfg.iam_user == "someone@example.com"
    assert cfg.embedding_dim == 768


def test_instance_connection_name_property():
    cfg = _build_pgvector_config(
        {"project_id": "my-proj", "region": "us-east1", "instance_name": "my-inst"}
    )

    assert cfg.instance_connection_name == "my-proj:us-east1:my-inst"


def test_load_config_populates_pgvector_section():
    app_config = load_config("config/config.yaml")

    pgvector = app_config.indexing.pgvector
    assert app_config.indexing.vector_index == "faiss"
    assert pgvector.enabled_for_build is False
    assert pgvector.project_id == "rag-with-receipts"
    assert pgvector.region == "northamerica-northeast1"
    assert pgvector.instance_name == "rag-receipts-pgvector"
    assert pgvector.database == "rag_receipts"
    assert pgvector.table_name == "chunks"
    assert pgvector.iam_user == "daniel.lofeodo@gmail.com"
    assert pgvector.embedding_dim == 1024
