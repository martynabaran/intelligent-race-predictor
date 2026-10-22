"""Load exports into a canonical DataFrame."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from race_predictor.domain import Activity
from race_predictor.exceptions import UnsupportedExportFormatError
from race_predictor.sources.base import ActivitySource, detect_source

COLUMNS = [
    "athlete",
    "date",
    "type",
    "distance_m",
    "elapsed_s",
    "moving_s",
    "avg_speed_ms",
    "max_speed_ms",
    "avg_hr",
    "max_hr",
    "elevation_gain_m",
    "elevation_loss_m",
    "avg_temp_c",
]


def activities_to_frame(activities: list[Activity]) -> pd.DataFrame:
    """Convert activities to a DataFrame with the canonical :data:`COLUMNS`."""
    df = pd.DataFrame([asdict(activity) for activity in activities])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.rename(columns={"athlete_id": "athlete"})
    df["type"] = (
        df["type"]
        .map(lambda activity_type: getattr(activity_type, "value", activity_type))
        .astype(object)
    )
    df["date"] = pd.to_datetime(df["date"])
    return df[COLUMNS]


def load_activities(
    path: str | Path, athlete_id: str | None = None, source: ActivitySource | None = None
) -> pd.DataFrame:
    """Load one export file (format auto-detected unless ``source`` is given)."""
    path = Path(path)
    source = source or detect_source(path)
    athlete_id = athlete_id or f"{source.name}_{path.stem}"
    return activities_to_frame(list(source.read(path, athlete_id)))


def load_directory(directory: str | Path) -> pd.DataFrame:
    """Load every CSV in ``directory`` as a separate athlete; unsupported files are skipped."""
    frames = []
    for index, path in enumerate(sorted(Path(directory).glob("*.csv")), start=1):
        try:
            source = detect_source(path)
        except UnsupportedExportFormatError:
            continue
        frames.append(load_activities(path, f"{source.name}_{index}", source))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
