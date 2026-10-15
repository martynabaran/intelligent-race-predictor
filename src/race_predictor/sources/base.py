"""Activity source abstraction and registry (Adapter + Registry patterns)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from race_predictor.domain import Activity
from race_predictor.exceptions import UnsupportedExportFormatError

_REGISTRY: dict[str, type[ActivitySource]] = {}


class ActivitySource(ABC):
    """Reads one export file and yields canonical :class:`Activity` objects."""

    name: str

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Cheap check (header sniffing) whether ``path`` is in this source's format."""

    @abstractmethod
    def read(self, path: Path, athlete_id: str) -> Iterator[Activity]:
        """Lazily yield activities from ``path``."""


def register_source(name: str) -> Callable[[type[ActivitySource]], type[ActivitySource]]:
    """Class decorator registering a source under ``name``."""

    def decorator(cls: type[ActivitySource]) -> type[ActivitySource]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_sources() -> list[str]:
    return sorted(_REGISTRY)


def get_source(name: str) -> ActivitySource:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise UnsupportedExportFormatError(
            f"Unknown source {name!r}; available: {available_sources()}"
        ) from None


def detect_source(path: str | Path) -> ActivitySource:
    """Pick the registered source that recognises ``path``."""
    path = Path(path)
    for cls in _REGISTRY.values():
        if cls.can_read(path):
            return cls()
    raise UnsupportedExportFormatError(f"{path.name} is not a supported Strava/Garmin export")


def read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return []


def to_float(value: object) -> float:
    """Parse numbers like ``"6,246"``, ``"--"`` or ``None`` into a float (NaN on failure)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).strip().replace(",", ".")
    if text in {"", "--", "None", "nan"}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan
