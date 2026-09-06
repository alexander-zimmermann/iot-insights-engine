"""Runner-lifecycle tests, through the injected store and publisher ends.

The guarantees the hand-written runners never had tests for live here:
the publish goes out before the write, a dry run touches neither write
side, a missing frontier stops the run, and the measurement sees the same
open rows the reconciliation uses. Fakes sit at both seams — no cluster,
no live database.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from iot_insights_engine.episode_store import OpenEpisodeRow
from iot_insights_engine.episodes import Episode, Observation
from iot_insights_engine.faults import Fault, MeasurementKind, Target
from iot_insights_engine.reconcile import Measured, Window
from iot_insights_engine.runner import LOOKBACK, Kind, run_subjects

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

_T0 = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)
_FRONTIER = _T0 + 8 * timedelta(hours=1)

_Applied = tuple[
    tuple[Episode, ...], tuple[tuple[int, Episode], ...], tuple[tuple[int, datetime], ...]
]


class _Store:
    """In-memory store end; every interaction lands in the shared event log."""

    def __init__(
        self, events: list[str], open_rows: list[OpenEpisodeRow] | None = None
    ) -> None:
        self.events = events
        self._open_rows = open_rows or []
        self.applied: _Applied | None = None

    @contextmanager
    def read(self) -> Iterator[Any]:
        self.events.append("read")
        yield None

    def open_rows(self, _conn: Any, _fault_name: str) -> list[OpenEpisodeRow]:
        return list(self._open_rows)

    def history_scores(self, _conn: Any, _fault_name: str) -> list[float]:
        return []

    def apply(
        self,
        _fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
    ) -> None:
        self.events.append("apply")
        self.applied = (tuple(inserts), tuple(updates), tuple(orphan_closes))


class _Publisher:
    """In-memory publisher end sharing the store's event log."""

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.published: list[tuple[str, str | None, dict[str, Any], str | None, bool]] = []

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None,
        firing: bool,
    ) -> None:
        self.events.append("publish")
        self.published.append((fault_name, severity, payload, entity, firing))


@dataclass(frozen=True, slots=True)
class _Publish:
    subject: str
    severity: int

    @property
    def entity(self) -> str:
        return self.subject


def _fault(target: Target | None = None) -> Fault:
    return Fault(
        name="test_fault",
        sentence="a device runs longer than declared",
        unit="h",
        kind=MeasurementKind.DURATION,
        parameters={},
        target=target if target is not None else Target(per_device=True),
    )


def _kind(
    *,
    frontier: datetime | None = _FRONTIER,
    observations: tuple[Observation, ...] = (),
    seen: dict[str, Any] | None = None,
) -> Kind[Any, Any]:
    def measure(
        _conn: Any, _fault: Fault, window: Window, open_rows: Sequence[OpenEpisodeRow]
    ) -> Measured[Any]:
        if seen is not None:
            seen["window"] = window
            seen["open_rows"] = tuple(open_rows)
        return Measured(
            states={}, observations=observations, dataless=frozenset(), counts={"subjects": 1}
        )

    return Kind(
        event="test_kind_run",
        delivery="per_device",
        frontier=lambda _conn: frontier,
        measure=measure,
        publish_for=lambda subject, severity, _state: _Publish(subject, severity),
        payload=lambda p: {"subject": p.subject},
    )


_FIRING = (Observation(subject="2/1/197", time=_FRONTIER, score=1.0),)


def test_the_publish_goes_out_before_the_write() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=False)

    assert events == ["read", "publish", "apply"]
    ((fault_name, severity, payload, entity, firing),) = publisher.published
    assert fault_name == "test_fault"
    assert severity is not None
    assert payload == {"subject": "2/1/197"}
    assert entity == "2/1/197"
    assert firing is True
    assert store.applied is not None
    inserts, _, _ = store.applied
    assert [e.subject for e in inserts] == ["2/1/197"]


def test_a_clear_is_published_before_its_row_closes() -> None:
    events: list[str] = []
    row = OpenEpisodeRow(id=7, subject="2/1/197", severity=2)
    store, publisher = _Store(events, open_rows=[row]), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(), dry_run=False)

    assert events == ["read", "publish", "apply"]
    ((_, severity, _, _, firing),) = publisher.published
    assert severity is None
    assert firing is False
    assert store.applied is not None
    _, _, orphan_closes = store.applied
    assert orphan_closes == ((7, _FRONTIER),)


def test_a_dry_run_touches_neither_write_side() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    run_subjects(store, publisher, _fault(), _kind(observations=_FIRING), dry_run=True)

    assert events == ["read"]
    assert publisher.published == []
    assert store.applied is None


def test_a_missing_frontier_stops_the_run_before_the_measurement() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)
    seen: dict[str, Any] = {}

    run_subjects(store, publisher, _fault(), _kind(frontier=None, seen=seen), dry_run=False)

    assert events == ["read"]
    assert seen == {}
    assert publisher.published == []
    assert store.applied is None


def test_a_fault_declaring_the_wrong_target_form_is_rejected() -> None:
    events: list[str] = []
    store, publisher = _Store(events), _Publisher(events)

    with pytest.raises(ValueError, match="per_device"):
        run_subjects(
            store, publisher, _fault(target=Target(ga="0/0/230")), _kind(), dry_run=False
        )
    assert events == []


def test_the_measurement_sees_the_window_and_the_open_rows() -> None:
    events: list[str] = []
    row = OpenEpisodeRow(id=7, subject="2/1/197", severity=2)
    store, publisher = _Store(events, open_rows=[row]), _Publisher(events)
    seen: dict[str, Any] = {}

    run_subjects(
        store, publisher, _fault(), _kind(observations=_FIRING, seen=seen), dry_run=False
    )

    assert seen["open_rows"] == (row,)
    assert seen["window"].frontier == _FRONTIER
    assert seen["window"].start == _FRONTIER - LOOKBACK
