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
    """Either one group address, or one address per matched main group in
    that group's Zentral block (channel silence) — resolved via the catalog.
    """

    ga: str | None = None
    per_main_group: bool = False

    def __post_init__(self) -> None:
        # Mirrors the schema's oneOf so the union holds for Python-side
        # construction too, not only for loaded files.
        if (self.ga is not None) == self.per_main_group:
            raise ValueError("target is either one ga or per_main_group, never both or neither")


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
    parameters expressed in the channel's own unit.
    """

    name: str
    sentence: str
    unit: str
    kind: MeasurementKind
    parameters: Mapping[str, float]
    scope: Scope
    target: Target
    dormant: Dormant | None = None


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


def _parse_fault(raw: dict[str, Any]) -> Fault:
    scope = raw["scope"]
    target = raw["target"]
    dormant = raw.get("dormant")
    return Fault(
        name=raw["name"],
        sentence=raw["sentence"],
        unit=raw["unit"],
        kind=MeasurementKind(raw["kind"]),
        parameters=MappingProxyType(dict(raw["parameters"])),
        scope=Scope(
            dpt=_as_tuple(scope.get("dpt")),
            name_like=_as_tuple(scope.get("name_like")),
            exclude_name_like=_as_tuple(scope.get("exclude_name_like")),
        ),
        target=Target(
            ga=target.get("ga"),
            per_main_group=target.get("per_main_group", False),
        ),
        dormant=(
            Dormant(reason=dormant["reason"], active_when=dormant["active_when"])
            if dormant is not None
            else None
        ),
    )
