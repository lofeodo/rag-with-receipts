from rag_receipts.config import load_config
from rag_receipts.eval.config import _build_eval_config


def test_defaults_when_section_empty():
    cfg = _build_eval_config({})

    assert cfg.judge_model == "claude-sonnet-5"
    assert cfg.judge_max_tokens == 1024
    assert cfg.dataset_path == "data/eval/qa_pairs.json"
    assert cfg.output_path == "results/eval_report.json"


def test_overrides_are_applied():
    raw = {
        "judge_model": "claude-opus-5-5",
        "judge_max_tokens": 2048,
        "dataset_path": "data/eval/other_pairs.json",
        "output_path": "results/other_report.json",
    }

    cfg = _build_eval_config(raw)

    assert cfg.judge_model == "claude-opus-5-5"
    assert cfg.judge_max_tokens == 2048
    assert cfg.dataset_path == "data/eval/other_pairs.json"
    assert cfg.output_path == "results/other_report.json"


def test_load_config_populates_eval_section():
    app_config = load_config("config/config.yaml")

    assert app_config.eval.judge_model == "claude-sonnet-5"
    assert app_config.eval.judge_max_tokens == 1024
    assert app_config.eval.dataset_path == "data/eval/qa_pairs.json"
    assert app_config.eval.output_path == "results/eval_report.json"
