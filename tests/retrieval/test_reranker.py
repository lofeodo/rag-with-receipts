import numpy as np

from .helpers import FakeCrossEncoder

from rag_receipts.retrieval.config import RerankerConfig
from rag_receipts.retrieval.reranker import Reranker, rerank_text


def test_rerank_text_includes_breadcrumb():
    assert rerank_text("Test page", "Section A", "some text") == (
        "Test page > Section A\n\nsome text"
    )


def test_rerank_text_no_section_path():
    assert rerank_text("Test page", "", "some text") == "Test page\n\nsome text"


def test_score_builds_query_passage_pairs():
    fake = FakeCrossEncoder()
    reranker = Reranker(config=RerankerConfig(), model=fake)

    reranker.score("my query", ["passage one", "passage two"])

    assert fake.calls == [[("my query", "passage one"), ("my query", "passage two")]]


class _SpyCrossEncoder:
    def __init__(self):
        self.kwargs = None

    def predict(self, sentences, *, batch_size, show_progress_bar):
        self.kwargs = {"batch_size": batch_size, "show_progress_bar": show_progress_bar}
        return np.zeros(len(sentences), dtype="float32")


def test_score_passes_config_through():
    spy = _SpyCrossEncoder()
    config = RerankerConfig(batch_size=8, show_progress_bar=True)
    reranker = Reranker(config=config, model=spy)

    reranker.score("q", ["p"])

    assert spy.kwargs == {"batch_size": 8, "show_progress_bar": True}


def test_score_matches_fake_overlap_function():
    fake = FakeCrossEncoder()
    reranker = Reranker(config=RerankerConfig(), model=fake)

    scores = reranker.score("abyssal whip special attack", [
        "the abyssal whip has a special attack",
        "slayer training uses a slayer master",
    ])

    assert scores.dtype.name == "float32"
    assert scores[0] > scores[1]


def test_from_config_wires_up_model(monkeypatch):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(
        "rag_receipts.retrieval.reranker.get_reranker_model", lambda model_name, device: fake
    )

    reranker = Reranker.from_config(RerankerConfig(device="cpu"))

    assert reranker.model is fake
