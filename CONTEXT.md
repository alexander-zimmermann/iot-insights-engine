# Domain model

The vocabulary this codebase is written in. Terms here are the names used
in module docstrings, log records and tests — a new module should reach
for one of these before inventing its own.

## Fault

One declared thing that can be wrong with the house, written as a
sentence with a unit in the fault file (`faults.yaml`, kept in the lares
repo beside the GA catalog and the writer rules). A fault names what it
measures (its **scope**), how (its **kind**), the parameters it measures
by, and where the verdict is **delivered**. The loader validates the file
against a bundled JSON Schema and freezes it into dataclasses, so a bad
edit fails at load, never at runtime in the cluster.

A **dormant** fault loads fully but is excluded from the schedule: it
declares why it cannot run yet and the observable condition under which it
starts to.

## Kind

How a fault is measured. Every fault has exactly one:

- **silence** — a channel that used to send has gone quiet, longer than a
  multiple of its own normal pause;
- **duration** — a device draws current for longer than its declared
  limit;
- **drift** — something a device does sits persistently above (or, for
  recovery, below) the healthy level declared for it, walked as a CUSUM
  with a pinned reference. Which series it walks is the fault's declared
  **signal**: `standby`, `duty_cycle` or `recovery`;
- **deviation** — a value sits too far under its declared reference. Two
  shapes, told apart by what the fault declares that reference as: rooms
  against a channel of the house while a gate condition holds, or the
  plant's daily yield against a model named in the entry — its declared
  **expectation**, swappable there without touching the comparison;
- **volume** — more than N incidents in a week, measured over the engine's
  own episode stream;
- **external** — Basalte detects and delivers the fault itself; the engine
  reads its severity writes back off the bus archive and only records.

`runner.Kind` is the code-level declaration of one: what it measures, how
its payload is shaped, and — where the defaults do not fit — how it folds
and plans.

## Subject

The thing a fault's verdict is about: a channel, a device, a room, an
exchanger, or the house itself. Episodes are stored per fault and subject.

## Scope

A fault's channel query, resolved where the catalog lives (`ga_catalog` in
TSDB) — never a hand-written address list. What it resolves to is a
**channel**: a group address, its catalog name and its DPT.

## Observation

One per-bucket measurement of a firing fault on one subject, carrying a
**score**: the fault's own magnitude in its declared unit, compared only
against that fault's own history, never across faults.

## Episode

One incident: when it started, when it was last seen, how bad it got, with
the per-bucket evidence rows that formed it and at most three
**notification events** (appearing, escalating, ending). Episodes are the
only stored artifact — everything else is recomputed from history on every
run, so a redeploy cannot corrupt or lose state.

## Fold

Turning repeated observations into episodes: the pure seam
`episodes.fold_observations`. External is the one kind that folds
something else (severity writes).

## Severity

A tier 0–3 (clear, info, warning, critical) — the delivery contract with
Basalte. Within an episode the tier is the quantile of the fault's own
score distribution, promoted one step by duration; there is no global
ladder. A stored severity is never lowered.

## Reconciliation

The computed episodes against the open episode rows the database already
holds: what to insert, what to update, which orphaned rows to close, and
which subjects moved to a new severity. A subject is **dataless** when
this run cannot tell a recovery from a blind spot — its measurement does
not reach the frontier — and its episode then stays open instead of
self-clearing.

## Frontier

The "now" a run is measured against: the newest bucket of the aggregate it
reads, not the wall clock — for a kind that measures whole days, the newest
*complete* one. The continuous aggregate materializes with an
end offset, so the newest visible bucket lags real time for every channel
at once; the frontier cancels that lag. Episodes also *end* in frontier
time, so a stalled refresh freezes the picture instead of clearing every
open episode with a severity nobody earned. External faults are the
exception — the bus archive has no materialization lag, so they declare
wall-clock time.

## Delivery

How a verdict leaves the engine: one publish per moved subject on
`anomaly.<fault>[.<entity>]`, carrying a numeric `severity_level` the
knx-nats-bridge writer rules route to a KNX group address, where Basalte
owns the text. The subject shape is pinned by those writer rules. A fault
declares its **target** — one address, one per main group, one per device,
one per room — and the kind's declaration says which form it expects.

## Runner

The run lifecycle every kind shares: window off the frontier, fold in the
kind's own cadence, reconciliation, run record, dry-run gating, and the
publish-before-write tail. It touches the world through two injected ends, the **store** and
the **publisher**, so those guarantees are testable through fakes.
Publishes go out before the database writes: a failed run then repeats the
same publish instead of losing it behind an already-updated database.
