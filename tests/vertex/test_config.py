from rag_receipts.config import load_config
from rag_receipts.vertex.config import _build_vertex_config


def test_defaults_when_section_empty():
    cfg = _build_vertex_config({})

    assert cfg.project_id == ""
    assert cfg.location == "global"
    assert cfg.data_store_id == "rag-receipts-corpus"
    assert cfg.engine_id == "rag-receipts-search-app"
    assert cfg.search_serving_config == "default_search"
    assert cfg.answer_serving_config == "default_serving_config"
    assert cfg.search_top_k == 10
    assert cfg.gcs_bucket == ""
    assert cfg.gcs_prefix == "corpus"
    assert cfg.search_results_path == "results/vertex_search_eval.json"
    assert cfg.answer_results_path == "results/vertex_answer_eval.json"


def test_overrides_are_applied():
    raw = {
        "project_id": "my-project",
        "location": "us-central1",
        "data_store_id": "my-store",
        "engine_id": "my-engine",
        "search_serving_config": "custom_search",
        "answer_serving_config": "custom_serving",
        "search_top_k": 20,
        "gcs_bucket": "my-bucket",
        "gcs_prefix": "data",
        "search_results_path": "results/custom_search.json",
        "answer_results_path": "results/custom_answer.json",
    }

    cfg = _build_vertex_config(raw)

    assert cfg.project_id == "my-project"
    assert cfg.location == "us-central1"
    assert cfg.data_store_id == "my-store"
    assert cfg.engine_id == "my-engine"
    assert cfg.search_serving_config == "custom_search"
    assert cfg.answer_serving_config == "custom_serving"
    assert cfg.search_top_k == 20
    assert cfg.gcs_bucket == "my-bucket"
    assert cfg.gcs_prefix == "data"
    assert cfg.search_results_path == "results/custom_search.json"
    assert cfg.answer_results_path == "results/custom_answer.json"


def test_load_config_populates_vertex_section():
    app_config = load_config("config/config.yaml")

    assert app_config.vertex.location == "global"
    assert app_config.vertex.data_store_id == "rag-receipts-corpus"
    assert app_config.vertex.engine_id == "rag-receipts-search-app"
    assert app_config.vertex.search_top_k == 10
    assert app_config.vertex.gcs_prefix == "corpus"
