"""Run-preparation tests — the shared step of the fault list.

Every test feeds an invented series of active buckets and asserts only the
runs that come out: where they start, where they end, how long they are.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from iot_insights_engine.runs import Run, split_runs

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_HOUR = timedelta(hours=1)


def _hours(*offsets: int) -> list[datetime]:
    return [_T0 + n * _HOUR for n in offsets]


def test_consecutive_buckets_form_one_run() -> None:
    assert split_runs(_hours(0, 1, 2, 3), _HOUR) == (
        Run(start=_T0, end=_T0 + 3 * _HOUR, duration=4 * _HOUR),
    )


def test_missing_bucket_splits_runs() -> None:
    assert split_runs(_hours(0, 1, 3, 4), _HOUR) == (
        Run(start=_T0, end=_T0 + _HOUR, duration=2 * _HOUR),
        Run(start=_T0 + 3 * _HOUR, end=_T0 + 4 * _HOUR, duration=2 * _HOUR),
    )


def test_single_bucket_is_a_run_of_one_bucket() -> None:
    assert split_runs(_hours(5), _HOUR) == (
        Run(start=_T0 + 5 * _HOUR, end=_T0 + 5 * _HOUR, duration=_HOUR),
    )


def test_no_buckets_no_runs() -> None:
    assert split_runs([], _HOUR) == ()


def test_input_order_does_not_matter() -> None:
    assert split_runs(_hours(3, 0, 1), _HOUR) == (
        Run(start=_T0, end=_T0 + _HOUR, duration=2 * _HOUR),
        Run(start=_T0 + 3 * _HOUR, end=_T0 + 3 * _HOUR, duration=_HOUR),
    )


def test_other_cadences_work() -> None:
    # The step is cadence-agnostic: freezer icing will feed minute samples.
    minute = timedelta(minutes=1)
    times = [_T0 + n * minute for n in (0, 1, 2, 10)]
    assert split_runs(times, minute) == (
        Run(start=_T0, end=_T0 + 2 * minute, duration=3 * minute),
        Run(start=_T0 + 10 * minute, end=_T0 + 10 * minute, duration=minute),
    )
