"""Top-level AppConfig: assembles each step's sub-config from config.yaml.

Each step owns its own config dataclasses + a `_build_<step>_config(raw) -> StepConfig`
helper in that step's package (e.g. `ingestion/config.py`, `indexing/config.py`). This
module only owns the top-level shape and the YAML read, so no step's package needs to
import another step's package just to load a config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag_receipts.eval.config import EvalConfig, _build_eval_config
from rag_receipts.generation.config import GenerationConfig, _build_generation_config
from rag_receipts.grounding.config import GroundingConfig, _build_grounding_config
from rag_receipts.indexing.config import IndexingConfig, _build_indexing_config
from rag_receipts.ingestion.config import CorpusConfig, IngestionConfig, _build_ingestion_config
from rag_receipts.retrieval.config import RetrievalConfig, _build_retrieval_config
from rag_receipts.vertex.config import VertexConfig, _build_vertex_config


@dataclass
class AppConfig:
    corpus: CorpusConfig
    ingestion: IngestionConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    grounding: GroundingConfig = field(default_factory=GroundingConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    corpus = CorpusConfig(**raw["corpus"])
    ingestion = _build_ingestion_config(raw["ingestion"])
    indexing = _build_indexing_config(raw.get("indexing", {}))
    retrieval = _build_retrieval_config(raw.get("retrieval", {}))
    generation = _build_generation_config(raw.get("generation", {}))
    eval_config = _build_eval_config(raw.get("eval", {}))
    grounding = _build_grounding_config(raw.get("grounding", {}))
    vertex = _build_vertex_config(raw.get("vertex", {}))
    return AppConfig(
        corpus=corpus,
        ingestion=ingestion,
        indexing=indexing,
        retrieval=retrieval,
        generation=generation,
        eval=eval_config,
        grounding=grounding,
        vertex=vertex,
    )
