"""Stage-timing utilities (Step 7): a stopwatch context manager and pure
percentile/summary functions, used by scripts/measure_latency.py and
scripts/sweep_reranker.py. No model, no I/O - fully unit-testable on plain floats.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy.percentile's default method.

    Implemented directly (rather than importing numpy) since this module has no
    other reason to depend on it.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (p / 100) * (n - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    frac = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * frac


def summarize_durations(durations: list[float]) -> dict:
    if not durations:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "count": 0}
    return {
        "p50": percentile(durations, 50),
        "p95": percentile(durations, 95),
        "mean": sum(durations) / len(durations),
        "count": len(durations),
    }


@dataclass
class Stopwatch:
    """with Stopwatch() as sw: ... ; sw.elapsed_seconds afterward."""

    elapsed_seconds: float = field(default=0.0, init=False)
    _start: float = field(default=0.0, init=False)

    def __enter__(self) -> "Stopwatch":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.elapsed_seconds = time.perf_counter() - self._start
        return False
