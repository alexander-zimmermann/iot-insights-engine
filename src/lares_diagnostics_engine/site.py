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

from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .declared import SCHEMAS, load_declared

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA_PATH = SCHEMAS / "site.schema.json"


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
        data = load_declared(path, _SCHEMA_PATH)

        timezone = data["timezone"]
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"{path}: timezone: unknown zone {timezone!r}") from exc

        planes = tuple(
            Plane(
                key=key,
                inverter_id=raw["inverter_id"],
                tilt=raw["tilt"],
                azimuth=raw["azimuth"],
                kwp=raw["kwp"],
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
                latitude=data["location"]["latitude"],
                longitude=data["location"]["longitude"],
            ),
            timezone=timezone,
            planes=planes,
        )
