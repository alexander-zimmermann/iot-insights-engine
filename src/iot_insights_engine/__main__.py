from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from .config import load_settings
from .logging_setup import configure_logging, get_logger

# Each subcommand maps to a sibling module `iot_insights_engine.<module>`
# that exposes `def run(settings, argv) -> int`. The dispatcher fails
# with a clear message if a subcommand listed here has no module yet.
SUBCOMMANDS: tuple[str, ...] = (
    "detect-univariate",
    "train-iforest",
    "score-iforest",
    "detect-knx-join",
    "score-seasonal",
    "forecast-solar",
    "forecast-weather",
    "energy-balance",
    "weekly-report",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="iot-insights-engine")
    parser.add_argument("subcommand", choices=SUBCOMMANDS)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    settings = load_settings()
    configure_logging(settings.log_level, settings.log_format)
    log = get_logger(__name__)

    module_name = args.subcommand.replace("-", "_")
    try:
        module = importlib.import_module(f"iot_insights_engine.{module_name}")
    except ImportError as exc:
        log.error("subcommand_not_implemented", subcommand=args.subcommand, error=str(exc))
        return 2

    try:
        rc: int = module.run(settings, args.rest)
    except Exception:
        log.exception("subcommand_failed", subcommand=args.subcommand)
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
