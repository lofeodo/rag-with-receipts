from rag_receipts.config import load_config
from rag_receipts.generation.config import _build_generation_config


def test_defaults_when_section_empty():
    cfg = _build_generation_config({})

    assert cfg.provider == "anthropic"
    assert cfg.model == "claude-sonnet-5"
    assert cfg.max_tokens == 4096


def test_overrides_are_applied():
    raw = {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "max_tokens": 8192,
    }

    cfg = _build_generation_config(raw)

    assert cfg.model == "claude-sonnet-5-5"
    assert cfg.max_tokens == 8192


def test_load_config_populates_generation_section():
    app_config = load_config("config/config.yaml")

    assert app_config.generation.provider == "anthropic"
    assert app_config.generation.model == "claude-sonnet-5"
    assert app_config.generation.max_tokens == 4096
