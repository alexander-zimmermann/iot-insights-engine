"""Site loader: `site.yaml` -> the typed description of what the house is.

The fault list says what counts as wrong; the site file says what is
there: where the house stands, which timezone its days are counted in,
and the PV planes with the inverter each of them feeds and the peak power
it carries. It lives in lares beside the fault list and is mounted into
the jobs the same way; this loader validates it against the bundled JSON
Schema and freezes it into dataclasses, so a bad edit fails at load with
an error naming the field, never at runtime in the cluster.

A plane is keyed by its name as the anomaly addresses carry it (`West`,
`Ost`), so the key travels into payloads and writer rules unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import jsonschema
import yaml

_SCHEMA_PATH = Path(__file__).resolve().parent / "_schemas" / "site.schema.json"


@dataclass(frozen=True, slots=True)
class Location:
    latitude: float
    longitude: float


@dataclass(frozen=True, slots=True)
class Plane:
    """One roof plane: the inverter it feeds, its orientation in the
    Open-Meteo convention (0 south, negative east, positive west) and its
    rated peak power in kWp — the most the plane can physically make in an
    hour, which is what a counter rise is bounded by."""

    key: str
    inverter_id: int
    tilt: float
    azimuth: float
    kwp: float


@dataclass(frozen=True, slots=True)
class Site:
    location: Location
    timezone: str
    planes: tuple[Plane, ...]

    @classmethod
    def load(cls, path: Path) -> Site:
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
            field = ".".join(str(p) for p in error.absolute_path)
            prefix = f"{field}: " if field else ""
            raise ValueError(f"{path}: {prefix}{error.message}") from error

        timezone = str(data["timezone"])
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"{path}: timezone: unknown zone {timezone!r}") from exc

        planes = tuple(
            Plane(
                key=key,
                inverter_id=int(raw["inverter_id"]),
                tilt=float(raw["tilt"]),
                azimuth=float(raw["azimuth"]),
                kwp=float(raw["kwp"]),
            )
            for key, raw in data["pv"]["planes"].items()
        )
        by_inverter: dict[int, Plane] = {}
        for plane in planes:
            # A counter belongs to one inverter; two planes claiming it
            # would count the same rises twice.
            if (other := by_inverter.get(plane.inverter_id)) is not None:
                raise ValueError(
                    f"{path}: pv.planes: inverter {plane.inverter_id} is claimed by both "
                    f"{other.key!r} and {plane.key!r}"
                )
            by_inverter[plane.inverter_id] = plane
        return cls(
            location=Location(
                latitude=float(data["location"]["latitude"]),
                longitude=float(data["location"]["longitude"]),
            ),
            timezone=timezone,
            planes=planes,
        )
