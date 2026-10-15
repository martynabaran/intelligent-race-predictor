"""Importing this package registers all built-in sources."""

from race_predictor.sources import garmin, strava  # noqa: F401  (registration side effect)
from race_predictor.sources.base import (
    ActivitySource,
    available_sources,
    detect_source,
    get_source,
    register_source,
)
from race_predictor.sources.loader import load_activities, load_directory

__all__ = [
    "ActivitySource",
    "available_sources",
    "detect_source",
    "get_source",
    "load_activities",
    "load_directory",
    "register_source",
]
