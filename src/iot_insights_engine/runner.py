"""The run lifecycle every fault kind shares.

A `Kind` declares what genuinely differs between kinds — how its series is
measured and how its payload is shaped, both living in the kind's own
module — and `run_subjects` owns the rest: the window off the aggregate's
frontier, the fold behind the pure episode seam, the reconciliation against
the stored open rows, the run record, the dry run, and the publish-then-
write tail. A new kind declares those things and inherits all of it.

The runner touches the world through two ends, injected so the lifecycle's
guarantees are testable through fakes:

* the **store end** — one read connection for the measurement, the open
  rows and score history behind it, and the transactional apply of a
  plan's row changes;
* the **publisher end** — one anomaly publish per moved subject.

Two adapters sit at each seam: the database and NATS in production, an
in-memory recorder in the runner's tests.

The ordering is load-bearing. Time is the aggregate's frontier throughout —
episodes also *end* in frontier time, so a stalled refresh (or a dead
bridge) freezes the picture instead of clearing every open episode with a
severity 0 nobody earned. Publishes go out before the database writes: a
failed run then repeats the same publish (same value, Basalte's change
detector ignores it) instead of losing it behind an already-updated
database. `--dry-run` computes and logs everything and touches neither
write side.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from . import episode_store, nats_publisher
from .db_write import read_connection, write_connection
from .episodes import Episode, EpisodePolicy, fold_observations
from .logging_setup import get_logger
from .reconcile import Measured, Plan, SubjectPublish, Window, subject_plan
from .severity import severity_name

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    import psycopg
    from psycopg.rows import DictRow

    from .config import Settings
    from .episode_store import OpenEpisodeRow
    from .faults import Fault

log = get_logger(__name__)

# Measurement window: pause estimation, observation reconstruction and the
# score history all live inside it. Matches the 30 days the episode fold-in
# started the comparison basis with.
LOOKBACK = timedelta(days=30)


class Store(Protocol):
    """The store end: what the runner needs of episode persistence."""

    def read(self) -> AbstractContextManager[psycopg.Connection[DictRow]]:
        """One connection for the whole read phase — frontier, measurement,
        open rows and score history see the same snapshot."""
        ...

    def open_rows(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[OpenEpisodeRow]: ...

    def history_scores(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[float]: ...

    def apply(
        self,
        fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
    ) -> None:
        """One plan's row changes, in one transaction."""
        ...


class Publisher(Protocol):
    """The publisher end: one anomaly publish per moved subject."""

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None,
        firing: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class DbStore:
    """The store end's database adapter — thin by policy, the cluster smoke
    test covers the SQL."""

    settings: Settings

    def read(self) -> AbstractContextManager[psycopg.Connection[DictRow]]:
        return read_connection(self.settings)

    def open_rows(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[OpenEpisodeRow]:
        return episode_store.open_rows(conn, fault_name)

    def history_scores(
        self, conn: psycopg.Connection[DictRow], fault_name: str
    ) -> list[float]:
        return episode_store.history_scores(conn, fault_name)

    def apply(
        self,
        fault_name: str,
        inserts: Sequence[Episode],
        updates: Sequence[tuple[int, Episode]],
        orphan_closes: Sequence[tuple[int, datetime]],
    ) -> None:
        with write_connection(self.settings) as conn, conn.transaction():
            episode_store.apply(conn, fault_name, inserts, updates, orphan_closes)


@dataclass(frozen=True, slots=True)
class NatsPublisher:
    """The publisher end's NATS adapter."""

    settings: Settings

    def publish_anomaly(
        self,
        fault_name: str,
        severity: str | None,
        payload: dict[str, Any],
        *,
        entity: str | None = None,
        firing: bool = True,
    ) -> None:
        nats_publisher.publish_anomaly(
            self.settings, fault_name, severity, payload, entity=entity, firing=firing
        )


class PlanHook[S, P](Protocol):
    """A kind's own planning step: the computed episodes against the stored
    open rows, delivered the kind's way — declared only where the default
    per-subject delivery does not fit (silence reports per main group).
    """

    def __call__(
        self,
        *,
        episodes: Sequence[Episode],
        open_rows: Sequence[OpenEpisodeRow],
        measured: Measured[S],
        frontier: datetime,
    ) -> Plan[P]: ...


@dataclass(frozen=True, slots=True)
class Kind[S, P: SubjectPublish]:
    """One fault kind, reduced to what actually differs between them: how
    its series is measured (`frontier`, `measure`) and how its payload is
    shaped (`publish_for`, `payload`) — both live in the kind's own module,
    so the whole wire story of a kind is read in one place. `event` names
    its log record, `delivery` the target form the fault must declare for
    it.

    `measure` receives the fault's open rows beside the window: most kinds
    ignore them, a kind that prunes its scope by what is already open (or
    seeds its fold with stored severities) reads them instead of re-asking
    the store.

    Planning defaults to the shared per-subject delivery through
    `publish_for`; a kind delivered another way declares `plan` instead —
    one of the two is required. `warn_dataless` is off for the one kind
    whose dataless set is routinely huge and already accounted for.
    """

    event: str
    delivery: str
    frontier: Callable[[psycopg.Connection[DictRow]], datetime | None]
    measure: Callable[
        [psycopg.Connection[DictRow], Fault, Window, Sequence[OpenEpisodeRow]],
        Measured[S],
    ]
    payload: Callable[[P], dict[str, Any]]
    publish_for: Callable[[str, int, S | None], P] | None = None
    plan: PlanHook[S, P] | None = None
    warn_dataless: bool = True


def publish_subjects[P: SubjectPublish](
    publisher: Publisher,
    fault_name: str,
    publishes: Iterable[P],
    payload: Callable[[P], dict[str, Any]],
) -> None:
    """One publish per moved subject, on the subject's own address: the
    severity decides firing, the kind decides the rest of the payload.
    """
    for publish in publishes:
        firing = publish.severity > 0
        publisher.publish_anomaly(
            fault_name,
            severity_name(publish.severity) if firing else None,
            payload(publish),
            entity=publish.entity,
            firing=firing,
        )


def run_subjects[S, P: SubjectPublish](
    store: Store, publisher: Publisher, fault: Fault, kind: Kind[S, P], *, dry_run: bool
) -> None:
    """The one shape a per-subject kind runs in: guard the declaration, take
    the window off the aggregate, measure, fold, reconcile, log — then
    publish before writing.
    """
    if fault.target is None or fault.target.form != kind.delivery:
        raise ValueError(
            f"fault {fault.name}: {fault.kind} delivery needs a {kind.delivery} target"
        )
    policy = EpisodePolicy()

    with store.read() as conn:
        frontier = kind.frontier(conn)
        if frontier is None:
            log.warning("no_aggregate_data", fault=fault.name)
            return
        window = Window(start=frontier - LOOKBACK, frontier=frontier, policy=policy)
        open_rows = store.open_rows(conn, fault.name)
        measured = kind.measure(conn, fault, window, open_rows)
        history_scores = store.history_scores(conn, fault.name)

    if measured.dataless and kind.warn_dataless:
        # Never at info level: a subject nobody could measure is the one
        # thing that keeps an open episode from ever clearing itself.
        log.warning(
            "subjects_dataless", fault=fault.name, subjects=sorted(measured.dataless)
        )

    # `now` is the frontier: episode ends are decided by aggregate progress,
    # never by wall time racing ahead of a stalled materialization.
    episodes = fold_observations(
        fault.name, measured.observations, history_scores, policy, frontier
    )

    plan = _plan_for(kind, episodes, open_rows, measured, frontier)

    log.info(
        kind.event,
        fault=fault.name,
        frontier=frontier.isoformat(),
        **measured.counts,
        episodes=len(episodes),
        open_episodes=sum(1 for e in episodes if e.ended_at is None),
        inserts=len(plan.inserts),
        updates=len(plan.updates),
        orphan_closes=len(plan.orphan_closes),
        stale_opens=list(plan.stale_opens),
        publishes=len(plan.publishes),
        dry_run=dry_run,
    )

    if dry_run:
        log_dry_run(fault, episodes, plan, measured.labels)
        return

    publish_subjects(publisher, fault.name, plan.publishes, kind.payload)
    store.apply(fault.name, plan.inserts, plan.updates, plan.orphan_closes)


def _plan_for[S, P: SubjectPublish](
    kind: Kind[S, P],
    episodes: Sequence[Episode],
    open_rows: Sequence[OpenEpisodeRow],
    measured: Measured[S],
    frontier: datetime,
) -> Plan[P]:
    """The kind's own plan where it declares one, the shared per-subject
    delivery otherwise."""
    if kind.plan is not None:
        return kind.plan(
            episodes=episodes, open_rows=open_rows, measured=measured, frontier=frontier
        )
    publish_for = kind.publish_for
    if publish_for is None:
        raise ValueError(f"kind {kind.event}: declares neither publish_for nor plan")

    def payload(subject: str, severity: int) -> P:
        return publish_for(subject, severity, measured.states.get(subject))

    return subject_plan(
        episodes=episodes,
        open_rows=open_rows,
        dataless=measured.dataless,
        frontier=frontier,
        publish_for=payload,
    )


def log_dry_run[P: SubjectPublish](
    fault: Fault,
    episodes: Sequence[Episode],
    plan: Plan[P],
    labels: Mapping[str, str],
) -> None:
    """What the run would have done, by subject — the dry run's whole point,
    so it names each subject the way a human does where the kind knows it.
    """
    per_subject = Counter(e.subject for e in episodes)
    log.info(
        "dry_run_episodes",
        fault=fault.name,
        per_subject={
            labels.get(subject, subject): count
            for subject, count in sorted(per_subject.items())
        },
        open_subjects=sorted(e.subject for e in episodes if e.ended_at is None),
        would_publish=[
            {"subject": p.subject, "severity": p.severity} for p in plan.publishes
        ],
    )
