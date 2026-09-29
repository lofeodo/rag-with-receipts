import json

import pandas as pd
from helpers import make_chunks_df

from rag_receipts.indexing.faiss_index import build_flat_ip_index, load_index
from rag_receipts.indexing.output import write_index, write_metadata, write_stats


def test_write_index_round_trip(tmp_path):
    import numpy as np

    index = build_flat_ip_index(np.eye(3, dtype="float32"))

    path = write_index(index, tmp_path, "faiss.index")

    assert path == tmp_path / "faiss.index"
    assert path.exists()
    reloaded = load_index(path)
    assert reloaded.ntotal == 3


def test_write_metadata_round_trip(tmp_path):
    df = make_chunks_df(3)

    path = write_metadata(df, tmp_path, "index_metadata.parquet")

    assert path.exists()
    reloaded = pd.read_parquet(path)
    assert list(reloaded["chunk_id"]) == list(df["chunk_id"])


def test_write_stats_produces_indented_json(tmp_path):
    stats = {"chunk_count": 5, "embedding_dim": 1024}
    stats_path = tmp_path / "nested" / "index_stats.json"

    result_path = write_stats(stats, stats_path)

    assert result_path == stats_path
    content = stats_path.read_text(encoding="utf-8")
    assert json.loads(content) == stats
    assert content == json.dumps(stats, indent=2, ensure_ascii=False)
