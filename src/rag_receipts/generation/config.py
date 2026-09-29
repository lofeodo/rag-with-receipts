"""Config for the generation pipeline (Claude Sonnet 5, tool-use citations).

No `temperature` field - Sonnet 5 runs adaptive thinking by default, and the API
rejects sampling params (temperature/top_p/top_k) with a 400 whenever thinking is
active. Determinism comes from the forced tool schema + strict mode, not sampling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-5"
    max_tokens: int = 4096


def _build_generation_config(raw: dict) -> GenerationConfig:
    return GenerationConfig(
        provider=raw.get("provider", "anthropic"),
        model=raw.get("model", "claude-sonnet-5"),
        max_tokens=raw.get("max_tokens", 4096),
    )
