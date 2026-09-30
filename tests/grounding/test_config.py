from rag_receipts.config import load_config
from rag_receipts.grounding.config import _build_grounding_config


def test_defaults_when_section_empty():
    cfg = _build_grounding_config({})

    assert cfg.nli_model == "cross-encoder/nli-deberta-v3-base"
    assert cfg.nli_device == "auto"
    assert cfg.nli_batch_size == 32
    assert cfg.nli_show_progress_bar is False
    assert cfg.entailment_threshold == 0.5
    assert cfg.contradiction_threshold == 0.5
    assert cfg.overlap_threshold == 0.3


def test_overrides_are_applied():
    raw = {
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "nli_device": "cpu",
        "nli_batch_size": 16,
        "nli_show_progress_bar": True,
        "entailment_threshold": 0.7,
        "contradiction_threshold": 0.6,
        "overlap_threshold": 0.2,
    }

    cfg = _build_grounding_config(raw)

    assert cfg.nli_model == "cross-encoder/nli-deberta-v3-small"
    assert cfg.nli_device == "cpu"
    assert cfg.nli_batch_size == 16
    assert cfg.nli_show_progress_bar is True
    assert cfg.entailment_threshold == 0.7
    assert cfg.contradiction_threshold == 0.6
    assert cfg.overlap_threshold == 0.2


def test_load_config_populates_grounding_section():
    app_config = load_config("config/config.yaml")

    assert app_config.grounding.nli_model == "cross-encoder/nli-deberta-v3-base"
    assert app_config.grounding.entailment_threshold == 0.5
    assert app_config.grounding.contradiction_threshold == 0.5
    assert app_config.grounding.overlap_threshold == 0.3
