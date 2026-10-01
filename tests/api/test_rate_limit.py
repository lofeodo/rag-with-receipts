from __future__ import annotations

from rag_receipts.api.rate_limit import RateLimiter


def test_allows_up_to_max_requests_then_blocks():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False


def test_different_identities_tracked_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)

    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False
    assert limiter.allow("5.6.7.8") is True


def test_old_hits_outside_window_are_forgotten(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=10)
    clock = {"t": 1000.0}
    monkeypatch.setattr("rag_receipts.api.rate_limit.time.monotonic", lambda: clock["t"])

    assert limiter.allow("1.2.3.4") is True
    assert limiter.allow("1.2.3.4") is False

    clock["t"] += 11
    assert limiter.allow("1.2.3.4") is True
