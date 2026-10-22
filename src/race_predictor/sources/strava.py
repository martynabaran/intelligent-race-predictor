"""Strava ``activities.csv`` export adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from dateutil import parser as date_parser
from typing_extensions import override

from race_predictor.domain import Activity, ActivityType
from race_predictor.sources.base import ActivitySource, read_header, register_source, to_float

_TYPE_MAP = {
    "Run": ActivityType.RUN,
    "Virtual Run": ActivityType.RUN,
    "Ride": ActivityType.RIDE,
    "Virtual Ride": ActivityType.RIDE,
    "Swim": ActivityType.SWIM,
    "Weight Training": ActivityType.WEIGHT_TRAINING,
}
_REQUIRED = {"Activity Date", "Activity Type", "Activity Name", "Moving Time"}


def parse_strava_date(text: str) -> pd.Timestamp:
    """Parse e.g. ``"Jul 17, 2021, 3:01:35 PM"``; returns NaT when unparseable."""
    try:
        return pd.Timestamp(date_parser.parse(text, dayfirst=False))
    except (ValueError, OverflowError, TypeError):
        return pd.NaT  # type: ignore[return-value]  # NaT is the documented "no date" sentinel


@register_source("strava")
class StravaCsvSource(ActivitySource):
    """Reads the Strava ``activities.csv`` export."""

    @override
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return _REQUIRED.issubset(read_header(path))

    @override
    def read(self, path: Path, athlete_id: str) -> Iterator[Activity]:
        raw = pd.read_csv(path, low_memory=False)
        # Strava exports contain duplicated column names; pandas suffixes the 2nd copy with ".1".
        # Distance in column "Distance" is in km, the first "Max Heart Rate" is often empty.
        hr_max = raw["Max Heart Rate"]
        if "Max Heart Rate.1" in raw:
            hr_max = hr_max.combine_first(raw["Max Heart Rate.1"])
        hr_max_values = hr_max.to_numpy()

        for position, (_, row) in enumerate(raw.iterrows()):
            date = parse_strava_date(str(row["Activity Date"]))
            if pd.isna(date):
                continue
            yield Activity(
                athlete_id=athlete_id,
                date=date.to_pydatetime(),
                type=_TYPE_MAP.get(str(row["Activity Type"]), ActivityType.WORKOUT),
                distance_m=to_float(row.get("Distance")) * 1000,
                elapsed_s=to_float(row.get("Elapsed Time")),
                moving_s=to_float(row.get("Moving Time")),
                avg_speed_ms=to_float(row.get("Average Speed")),
                max_speed_ms=to_float(row.get("Max Speed")),
                avg_hr=to_float(row.get("Average Heart Rate")),
                max_hr=to_float(hr_max_values[position]),
                elevation_gain_m=to_float(row.get("Elevation Gain")),
                elevation_loss_m=to_float(row.get("Elevation Loss")),
                avg_temp_c=to_float(row.get("Average Temperature")),
            )
