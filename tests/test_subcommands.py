"""The engine's job surface.

With the old detection switched off, the forecast pulls, the energy
balance and the one fault runner are dispatchable — faults are YAML
entries behind `detect-faults`, never subcommands of their own.
"""

from __future__ import annotations

import importlib

from iot_insights_engine.__main__ import SUBCOMMANDS


def test_no_per_detector_jobs_remain() -> None:
    assert set(SUBCOMMANDS) == {
        "forecast-solar",
        "forecast-weather",
        "energy-balance",
        "detect-faults",
    }


def test_every_subcommand_resolves_to_a_module() -> None:
    for subcommand in SUBCOMMANDS:
        module = importlib.import_module(f"iot_insights_engine.{subcommand.replace('-', '_')}")
        assert callable(module.run)
