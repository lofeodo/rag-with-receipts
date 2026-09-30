from rag_receipts.config import load_config
from rag_receipts.retrieval.config import _build_retrieval_config


def test_defaults_when_section_empty():
    cfg = _build_retrieval_config({})

    assert cfg.top_k_dense == 30
    assert cfg.top_k_final == 5
    assert cfg.reranker.model_name == "BAAI/bge-reranker-base"
    assert cfg.reranker.batch_size == 32
    assert cfg.reranker.device == "auto"
    assert cfg.reranker.show_progress_bar is False


def test_overrides_are_applied():
    raw = {
        "top_k_dense": 50,
        "top_k_final": 8,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_batch_size": 16,
        "reranker_device": "cpu",
        "reranker_show_progress_bar": True,
    }

    cfg = _build_retrieval_config(raw)

    assert cfg.top_k_dense == 50
    assert cfg.top_k_final == 8
    assert cfg.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert cfg.reranker.batch_size == 16
    assert cfg.reranker.device == "cpu"
    assert cfg.reranker.show_progress_bar is True


def test_load_config_populates_retrieval_section():
    app_config = load_config("config/config.yaml")

    assert app_config.retrieval.top_k_dense == 30
    assert app_config.retrieval.top_k_final == 5
    assert app_config.retrieval.reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
