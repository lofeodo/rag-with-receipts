"""Config for the grounding/hallucination check (Step 6): an NLI cross-encoder +
lexical overlap combined into a grounded/ungrounded/contradicted verdict per citation.

Plain dataclass + PyYAML, matching retrieval/config.py and generation/config.py's style.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GroundingConfig:
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_device: str = "auto"  # auto | cpu | cuda
    nli_batch_size: int = 32
    nli_show_progress_bar: bool = False
    entailment_threshold: float = 0.5
    contradiction_threshold: float = 0.5
    overlap_threshold: float = 0.3


def _build_grounding_config(raw: dict) -> GroundingConfig:
    return GroundingConfig(
        nli_model=raw.get("nli_model", "cross-encoder/nli-deberta-v3-base"),
        nli_device=raw.get("nli_device", "auto"),
        nli_batch_size=raw.get("nli_batch_size", 32),
        nli_show_progress_bar=raw.get("nli_show_progress_bar", False),
        entailment_threshold=raw.get("entailment_threshold", 0.5),
        contradiction_threshold=raw.get("contradiction_threshold", 0.5),
        overlap_threshold=raw.get("overlap_threshold", 0.3),
    )
