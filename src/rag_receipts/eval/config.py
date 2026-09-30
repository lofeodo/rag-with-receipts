"""Config for the eval harness (Step 5): retrieval precision + LLM-as-judge correctness.

Plain dataclass + PyYAML, matching retrieval/config.py and generation/config.py's style.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalConfig:
    judge_model: str = "claude-sonnet-5"
    judge_max_tokens: int = 1024
    dataset_path: str = "data/eval/qa_pairs.json"
    output_path: str = "results/eval_report.json"


def _build_eval_config(raw: dict) -> EvalConfig:
    return EvalConfig(
        judge_model=raw.get("judge_model", "claude-sonnet-5"),
        judge_max_tokens=raw.get("judge_max_tokens", 1024),
        dataset_path=raw.get("dataset_path", "data/eval/qa_pairs.json"),
        output_path=raw.get("output_path", "results/eval_report.json"),
    )
