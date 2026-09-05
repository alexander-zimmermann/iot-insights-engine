"""Fault-list loader: YAML file -> validated, typed fault definitions.

The fault list lives in lares next to the GA catalog and the writer rules;
this loader validates it against the bundled JSON Schema and freezes it into
dataclasses — the same pattern the KNX bridge uses for its writer rules. A
bad edit fails here, at load, with an error naming the fault and the field,
never at runtime in the cluster.

The channel scope is a catalog query carried verbatim: the loader never
resolves it, so the file stays loadable without a catalog at hand. Dormant
faults load fully but are excluded from `schedulable()`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jsonschema
import yaml

_SCHEMA_PATH = Path(__file__).resolve().parent / "_schemas" / "faults.schema.json"


class MeasurementKind(StrEnum):
    DRIFT = "drift"
    DURATION = "duration"
    DEVIATION = "deviation"
    SILENCE = "silence"
    CONSTANCY = "constancy"
    # Not measured here: Basalte detects and delivers, the engine reads the
    # severity writes back off the fault address and records episodes.
    EXTERNAL = "external"
    # Not measured on channels at all: the count of incidents per week, over
    # the episode stream every other fault writes.
    VOLUME = "volume"


@dataclass(frozen=True, slots=True)
class Scope:
    """Catalog query text. Each field is a conjunctive criterion; list-valued
    criteria are disjunctive within themselves (any DPT of, any pattern of).
    Resolution — including dropping dead and dormant channels — happens where
    the catalog lives, never in the loader.
    """

    dpt: tuple[str, ...] = ()
    name_like: tuple[str, ...] = ()
    exclude_name_like: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Target:
    """One group address, one address per matched main group in that
    group's Zentral block (channel silence), one address per declared
    device (appliance runtime), or one per declared room (room deviation)
    — resolved via the catalog and the writer rules, never listed here.
    """

    ga: str | None = None
    per_main_group: bool = False
    per_device: bool = False
    per_room: bool = False

    def __post_init__(self) -> None:
        # Mirrors the schema's oneOf so the union holds for Python-side
        # construction too, not only for loaded files.
        forms = (self.ga is not None, self.per_main_group, self.per_device, self.per_room)
        if sum(forms) != 1:
            raise ValueError(
                "target is exactly one of ga, per_main_group, per_device or per_room"
            )

    @property
    def form(self) -> str:
        """Which of the four this target declares — exactly one, by
        construction, so a runner can compare it against the form its kind
        delivers on."""
        if self.ga is not None:
            return "ga"
        if self.per_main_group:
            return "per_main_group"
        if self.per_device:
            return "per_device"
        return "per_room"


@dataclass(frozen=True, slots=True)
class DeviceLimit:
    """One device's declared limit: a unique fragment of its catalog name,
    and the runtime it may not exceed — written here after someone looked
    at the data once, never derived from history.
    """

    match: str
    max_run_hours: float


@dataclass(frozen=True, slots=True)
class DeviceReference:
    """One device's declared healthy level: a unique fragment of its catalog
    name, and the standby draw it sits at when nothing is wrong — written
    here after someone read it off the data once, never derived from
    history. A reference the detector computed for itself would adopt a
    months-old fault as healthy, which is the whole reason it is declared.
    """

    match: str
    healthy_ma: float


@dataclass(frozen=True, slots=True)
class Roles:
    """The deviation kind's shared channel roles, each a catalog-name LIKE
    pattern matched inside every room's channels: the reference the value
    deviates from, and an optional gate that must stand for the fault to
    count at all. Both follow a uniform naming rule across rooms, which is
    what makes a pattern the honest way to write them; the measured value
    does not, so it is declared per room.
    """

    reference: str
    gate: str | None = None


@dataclass(frozen=True, slots=True)
class RoomRule:
    """One room's declared rule: a unique fragment of its catalog names, the
    channel it is measured on, and the gap under the reference it may not
    exceed — written here after someone looked at the data once. The value
    channel is named per room rather than matched by a shared pattern: rooms
    disagree about which channel is theirs, and naming it makes the file
    readable without resolving anything.
    """

    match: str
    min_gap_k: float
    value: str


@dataclass(frozen=True, slots=True)
class Dormant:
    """Declared dormancy: why the fault cannot run yet, and the observable
    condition under which it starts to — never a bare boolean with prose
    beside it.
    """

    reason: str
    active_when: str


@dataclass(frozen=True, slots=True)
class Fault:
    """One declared fault: a sentence with a unit, measured one way, with
    parameters expressed in the channel's own unit. External faults carry
    neither parameters nor a target — threshold and delivery live in
    Basalte, the scope names the address whose writes come back. A volume
    fault carries no scope: it measures the episode stream, not channels.
    """

    name: str
    sentence: str
    unit: str
    kind: MeasurementKind
    parameters: Mapping[str, float]
    scope: Scope | None = None
    target: Target | None = None
    dormant: Dormant | None = None
    devices: tuple[DeviceLimit, ...] = ()
    references: tuple[DeviceReference, ...] = ()
    roles: Roles | None = None
    rooms: tuple[RoomRule, ...] = ()

    def channel_scope(self) -> Scope:
        """The catalog query this fault measures over. Every kind but volume
        declares one and the loader enforces it, so a missing scope here is a
        new kind that arrived without saying what it measures — never an
        empty query, which would resolve to the whole catalog.
        """
        if self.scope is None:
            raise ValueError(f"fault {self.name}: this kind measures channels and needs a scope")
        return self.scope


class FaultList:
    def __init__(self, faults: list[Fault]) -> None:
        self._faults = tuple(faults)
        self._by_name = {f.name: f for f in faults}

    def __len__(self) -> int:
        return len(self._faults)

    def __iter__(self) -> Iterator[Fault]:
        return iter(self._faults)

    def get(self, name: str) -> Fault:
        return self._by_name[name]

    def schedulable(self) -> tuple[Fault, ...]:
        """The faults a scheduler may run — everything not declared dormant."""
        return tuple(f for f in self._faults if f.dormant is None)

    @classmethod
    def load(cls, path: Path) -> FaultList:
        raw_text = path.read_text(encoding="utf-8")
        data: Any = yaml.safe_load(raw_text) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a mapping at the top level, got {type(data).__name__}"
            )

        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        error = jsonschema.exceptions.best_match(validator.iter_errors(data))
        if error is not None:
            raise ValueError(f"{path}: {_describe(error, data)}") from error

        faults: list[Fault] = []
        seen: set[str] = set()
        for raw in data["faults"]:
            name = raw["name"]
            if name in seen:
                raise ValueError(f"{path}: duplicate fault name {name!r}")
            seen.add(name)
            problem = (
                _check_external(raw)
                or _check_volume(raw)
                or _check_devices(raw)
                or _check_references(raw)
                or _check_rooms(raw)
            )
            if problem is not None:
                raise ValueError(f"{path}: {problem}")
            faults.append(_parse_fault(raw))
        return cls(faults)


def _describe(error: jsonschema.ValidationError, data: Any) -> str:
    """Prefix the schema error with the fault it belongs to, by name where
    the entry has one, by position where the name itself is what is missing.
    """
    path = list(error.absolute_path)
    if len(path) >= 2 and path[0] == "faults" and isinstance(path[1], int):
        idx = path[1]
        raw = data["faults"][idx]
        name = raw.get("name") if isinstance(raw, dict) else None
        label = repr(name) if isinstance(name, str) else f"#{idx + 1}"
        field = ".".join(str(p) for p in path[2:])
        prefix = f"fault {label}: " + (f"{field}: " if field else "")
        return prefix + error.message
    return error.message


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _check_external(raw: dict[str, Any]) -> str | None:
    """External faults declare neither parameters nor a target: threshold
    and delivery live in Basalte, and a value here would be a second source
    of truth beside the Studio logic. Checked in the loader because the
    schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.EXTERNAL:
        return None
    for forbidden in ("parameters", "target"):
        if forbidden in raw:
            return (
                f"fault {raw['name']!r}: {forbidden}: "
                f"an external fault declares none — Basalte owns it"
            )
    return None


def _check_volume(raw: dict[str, Any]) -> str | None:
    """A volume fault measures the episode stream, so a channel scope on it
    would be configuration nothing reads. Checked in the loader because the
    schema's if/else error names the branch rather than the field.
    """
    if raw["kind"] == MeasurementKind.VOLUME and "scope" in raw:
        return (
            f"fault {raw['name']!r}: scope: a volume fault counts episodes, "
            f"not channels — it declares none"
        )
    return None


def _check_devices(raw: dict[str, Any]) -> str | None:
    """Per-device limits belong to the duration kind alone; anywhere else
    they would be dead configuration nothing reads. Checked in the loader
    because the schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.DURATION and "devices" in raw:
        return f"fault {raw['name']!r}: devices: only a duration fault declares per-device limits"
    return None


def _check_references(raw: dict[str, Any]) -> str | None:
    """Healthy references belong to the drift kind alone; anywhere else they
    would be dead configuration nothing reads. Checked in the loader because
    the schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.DRIFT and "references" in raw:
        return (
            f"fault {raw['name']!r}: references: "
            f"only a drift fault declares healthy references"
        )
    return None


def _check_rooms(raw: dict[str, Any]) -> str | None:
    """Roles and per-room rules belong to the deviation kind alone, and a
    gate role needs its threshold (and the other way round) — half a gate
    would silently measure ungated. Checked in the loader because the
    schema's false-subschema error loses the field name.
    """
    if raw["kind"] != MeasurementKind.DEVIATION:
        for field in ("roles", "rooms"):
            if field in raw:
                return (
                    f"fault {raw['name']!r}: {field}: "
                    f"only a deviation fault declares channel roles and rooms"
                )
        return None
    has_gate = "gate" in raw["roles"]
    has_gate_min = "gate_min_pct" in raw.get("parameters", {})
    if has_gate != has_gate_min:
        return (
            f"fault {raw['name']!r}: a gate role and the gate_min_pct parameter "
            f"come together — one without the other measures ungated"
        )
    return None


def _parse_fault(raw: dict[str, Any]) -> Fault:
    scope = raw.get("scope")
    target = raw.get("target")
    dormant = raw.get("dormant")
    roles = raw.get("roles")
    return Fault(
        name=raw["name"],
        sentence=raw["sentence"],
        unit=raw["unit"],
        kind=MeasurementKind(raw["kind"]),
        parameters=MappingProxyType(dict(raw.get("parameters", {}))),
        scope=(
            Scope(
                dpt=_as_tuple(scope.get("dpt")),
                name_like=_as_tuple(scope.get("name_like")),
                exclude_name_like=_as_tuple(scope.get("exclude_name_like")),
            )
            if scope is not None
            else None
        ),
        target=(
            Target(
                ga=target.get("ga"),
                per_main_group=target.get("per_main_group", False),
                per_device=target.get("per_device", False),
                per_room=target.get("per_room", False),
            )
            if target is not None
            else None
        ),
        devices=tuple(
            DeviceLimit(match=match, max_run_hours=limit["max_run_hours"])
            for match, limit in raw.get("devices", {}).items()
        ),
        references=tuple(
            DeviceReference(match=match, healthy_ma=reference["healthy_ma"])
            for match, reference in raw.get("references", {}).items()
        ),
        roles=(
            Roles(reference=roles["reference"], gate=roles.get("gate"))
            if roles is not None
            else None
        ),
        rooms=tuple(
            RoomRule(match=match, min_gap_k=rule["min_gap_k"], value=rule["value"])
            for match, rule in raw.get("rooms", {}).items()
        ),
        dormant=(
            Dormant(reason=dormant["reason"], active_when=dormant["active_when"])
            if dormant is not None
            else None
        ),
    )
