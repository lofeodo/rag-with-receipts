import numpy as np

from rag_receipts.indexing.faiss_index import build_flat_ip_index, load_index, save_index


def test_build_flat_ip_index_shape():
    embeddings = np.eye(4, dtype="float32")

    index = build_flat_ip_index(embeddings)

    assert index.ntotal == 4
    assert index.d == 4


def test_self_similarity_returns_top1_self():
    embeddings = np.eye(4, dtype="float32")
    index = build_flat_ip_index(embeddings)

    scores, ids = index.search(embeddings[1:2], 1)

    assert ids[0][0] == 1
    assert abs(scores[0][0] - 1.0) < 1e-5


def test_save_and_load_round_trip(tmp_path):
    embeddings = np.eye(4, dtype="float32")
    index = build_flat_ip_index(embeddings)
    path = tmp_path / "sub" / "faiss.index"

    save_index(index, path)
    reloaded = load_index(path)

    assert reloaded.ntotal == index.ntotal
    assert reloaded.d == index.d

    scores, ids = reloaded.search(embeddings[2:3], 1)
    assert ids[0][0] == 2
    assert abs(scores[0][0] - 1.0) < 1e-5
