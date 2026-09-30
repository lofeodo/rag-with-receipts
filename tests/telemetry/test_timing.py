from __future__ import annotations

import pytest

from rag_receipts.telemetry.timing import Stopwatch, percentile, summarize_durations


def test_percentile_median_of_odd_length():
    assert percentile([1, 2, 3, 4, 5], 50) == 3.0


def test_percentile_95_interpolates():
    # rank = 0.95 * 4 = 3.8 -> 4 + (5-4)*0.8 = 4.8, matches numpy.percentile's default.
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_percentile_single_value():
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_unsorted_input():
    assert percentile([5, 1, 3, 2, 4], 50) == 3.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_durations_basic():
    summary = summarize_durations([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["p50"] == 3.0
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["mean"] == 3.0
    assert summary["count"] == 5


def test_summarize_durations_empty():
    summary = summarize_durations([])
    assert summary == {"p50": 0.0, "p95": 0.0, "mean": 0.0, "count": 0}


def test_stopwatch_records_nonnegative_elapsed():
    with Stopwatch() as sw:
        pass
    assert isinstance(sw.elapsed_seconds, float)
    assert sw.elapsed_seconds >= 0.0


def test_stopwatch_does_not_suppress_exceptions():
    with pytest.raises(RuntimeError):
        with Stopwatch():
            raise RuntimeError("boom")
