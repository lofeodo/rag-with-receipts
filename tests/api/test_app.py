from __future__ import annotations

from fastapi.testclient import TestClient

from rag_receipts.api.app import create_app
from rag_receipts.api.rate_limit import RateLimiter
from rag_receipts.generation.models import GeneratedAnswer

from .helpers import FakeGenerator, FakeRetriever, RaisingGenerator


def _app(**overrides):
    kwargs = {"retriever": FakeRetriever(), "generator": FakeGenerator()}
    kwargs.update(overrides)
    return create_app(**kwargs)


def test_healthz_always_ok():
    client = TestClient(_app())
    resp = client.get("/healthz")
    assert resp.status_code == 200


def test_readyz_503_before_ready_then_200_after():
    app = _app()
    client = TestClient(app)
    assert client.get("/readyz").status_code == 503
    app.state.ready = True
    assert client.get("/readyz").status_code == 200


def test_query_503_when_not_ready():
    client = TestClient(_app())
    resp = client.post("/query", json={"query": "How do you start Monkey Madness I?"})
    assert resp.status_code == 503


def test_query_happy_path_resolves_citations_and_timing():
    app = _app()
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "What items are required to start Monkey Madness I?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answerable"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["chunk_id"] == "chunk_001"
    assert body["citations"][0]["page_title"] == "Monkey Madness I"
    assert body["hallucinated_citation_ids"] == []
    assert body["timing"]["total_s"] > 0


def test_query_hallucinated_citations_pass_through():
    bad_answer = GeneratedAnswer(
        query="q",
        answer="partial answer",
        citations=[],
        answerable=True,
        hallucinated_citation_ids=["ghost_chunk"],
        has_hallucinated_citations=True,
        model="claude-sonnet-5",
        input_tokens=10,
        output_tokens=5,
        stop_reason="tool_use",
    )
    app = _app(generator=FakeGenerator(answer=bad_answer))
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "anything"})

    assert resp.status_code == 200
    assert resp.json()["hallucinated_citation_ids"] == ["ghost_chunk"]


def test_query_not_answerable_passes_through():
    na_answer = GeneratedAnswer(
        query="q",
        answer="Not enough information in the retrieved context.",
        citations=[],
        answerable=False,
        hallucinated_citation_ids=[],
        has_hallucinated_citations=False,
        model="claude-sonnet-5",
        input_tokens=10,
        output_tokens=5,
        stop_reason="tool_use",
    )
    app = _app(generator=FakeGenerator(answer=na_answer))
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "Something out of corpus"})

    assert resp.status_code == 200
    assert resp.json()["answerable"] is False


def test_query_generator_runtime_error_returns_502():
    app = _app(generator=RaisingGenerator())
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "anything"})

    assert resp.status_code == 502


def test_query_rejects_empty_query():
    app = _app()
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": ""})

    assert resp.status_code == 422


def test_rate_limiter_429s_after_threshold():
    app = _app(rate_limiter=RateLimiter(max_requests=2, window_seconds=60))
    app.state.ready = True
    client = TestClient(app)

    for _ in range(2):
        assert client.post("/query", json={"query": "x"}).status_code == 200
    assert client.post("/query", json={"query": "x"}).status_code == 429


def test_query_401_when_demo_key_set_and_header_missing(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "shh-secret")
    app = _app()
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "x"})

    assert resp.status_code == 401


def test_query_401_when_demo_key_set_and_header_wrong(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "shh-secret")
    app = _app()
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "x"}, headers={"X-Demo-Key": "wrong"})

    assert resp.status_code == 401


def test_query_200_when_demo_key_set_and_header_correct(monkeypatch):
    monkeypatch.setenv("DEMO_API_KEY", "shh-secret")
    app = _app()
    app.state.ready = True
    client = TestClient(app)

    resp = client.post("/query", json={"query": "x"}, headers={"X-Demo-Key": "shh-secret"})

    assert resp.status_code == 200


def test_index_page_404_when_static_missing(tmp_path):
    app = _app(static_dir=tmp_path)
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 404


def test_index_page_served_when_present(tmp_path):
    (tmp_path / "index.html").write_text("<html>demo page</html>", encoding="utf-8")
    app = _app(static_dir=tmp_path)
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 200
    assert "demo page" in resp.text
