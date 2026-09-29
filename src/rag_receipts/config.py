"""Top-level AppConfig: assembles each step's sub-config from config.yaml.

Each step owns its own config dataclasses + a `_build_<step>_config(raw) -> StepConfig`
helper in that step's package (e.g. `ingestion/config.py`, `indexing/config.py`). This
module only owns the top-level shape and the YAML read, so no step's package needs to
import another step's package just to load a config file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_receipts.ingestion.config import CorpusConfig, IngestionConfig, _build_ingestion_config


@dataclass
class AppConfig:
    corpus: CorpusConfig
    ingestion: IngestionConfig


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    corpus = CorpusConfig(**raw["corpus"])
    ingestion = _build_ingestion_config(raw["ingestion"])
    return AppConfig(corpus=corpus, ingestion=ingestion)
