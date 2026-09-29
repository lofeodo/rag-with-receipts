from .helpers import FakeEncoder

from rag_receipts.indexing.config import EmbeddingConfig
from rag_receipts.indexing.embedder import Embedder, resolve_device


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_encode_passages_has_no_query_prefix():
    fake = FakeEncoder(dim=4)
    embedder = Embedder(config=EmbeddingConfig(), model=fake)

    result = embedder.encode_passages(["Abyssal whip", "Slayer training"])

    assert result.shape == (2, 4)
    assert result.dtype.name == "float32"
    assert fake.calls == [["Abyssal whip", "Slayer training"]]


def test_encode_queries_gets_instruction_prefix():
    fake = FakeEncoder(dim=4)
    config = EmbeddingConfig(query_instruction="PREFIX: ")
    embedder = Embedder(config=config, model=fake)

    embedder.encode_queries(["what is x?"])

    assert fake.calls == [["PREFIX: what is x?"]]


def test_from_config_wires_up_model(monkeypatch):
    fake = FakeEncoder(dim=4)
    monkeypatch.setattr(
        "rag_receipts.indexing.embedder.get_model", lambda model_name, device: fake
    )

    embedder = Embedder.from_config(EmbeddingConfig(device="cpu"))

    assert embedder.model is fake
