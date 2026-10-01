"""Per-identity in-memory sliding-window rate limiter (Step 8).

Deliberately simple: keyed on whatever identity string the caller resolves
(IAP's authenticated email, or client IP as a local-dev fallback), windows
tracked as a deque of timestamps per key. This is per-process state - it
resets on cold start and isn't shared across Cloud Run instances. That's an
accepted limitation (documented in CLAUDE.md), not a bug: at the deploy's
`--max-instances=2` cap on a low-traffic portfolio demo, it's enough to stop
one identity from hammering the real Anthropic API, not a production-grade
distributed limiter.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    max_requests: int
    window_seconds: float
    _hits: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))

    def allow(self, identity: str) -> bool:
        """Records this call as a hit and returns whether it's within the window's cap."""
        now = time.monotonic()
        hits = self._hits[identity]
        cutoff = now - self.window_seconds
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True
