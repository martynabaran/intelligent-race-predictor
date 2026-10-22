"""Garmin Connect CSV export adapter (Polish-locale headers, as in the thesis data)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from typing_extensions import override

from race_predictor.domain import Activity, ActivityType
from race_predictor.sources.base import ActivitySource, read_header, register_source, to_float

_TYPE_MAP = {
    "Bieganie": ActivityType.RUN,
    "Bieg na bieżni": ActivityType.RUN,
    "Virtual Run": ActivityType.RUN,
    "Run": ActivityType.RUN,
    "Kolarstwo": ActivityType.RIDE,
    "Kolarstwo szosowe": ActivityType.RIDE,
    "Jazda na rowerze treningowym": ActivityType.RIDE,
    "Jazda po drodze szutrowej/nieutwardzonej": ActivityType.RIDE,
    "Virtual Ride": ActivityType.RIDE,
    "Ride": ActivityType.RIDE,
    "Pływanie na basenie": ActivityType.SWIM,
    "Pływanie open water": ActivityType.SWIM,
    "Swim": ActivityType.SWIM,
    "Trening siłowy": ActivityType.WEIGHT_TRAINING,
    "Weight Training": ActivityType.WEIGHT_TRAINING,
}
_REQUIRED = {"Typ aktywności", "Data", "Dystans", "Czas"}
MAX_SWIM_DISTANCE_KM = 15.0  # larger swim distances in the export are in metres


def parse_duration(text: object) -> float:
    """Convert ``"HH:MM:SS(.f)"`` or ``"MM:SS"`` to seconds (NaN on failure)."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return math.nan
    parts = str(text).strip().split(":")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return math.nan
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def speed_to_ms(value: object, activity_type: ActivityType) -> float:
    """Convert Garmin speed/pace cells to m/s.

    Runs are stored as pace (``min/km``), swims as ``min/100 m``, rides as km/h.
    """
    text = str(value).strip().replace(",", ".")
    if text in {"", "--", "nan", "None"}:
        return math.nan
    try:
        if activity_type is ActivityType.RIDE:
            return float(text) / 3.6
        minutes = parse_duration(text) / 60 if ":" in text else float(text)
        if minutes <= 0:
            return math.nan
        if activity_type is ActivityType.RUN:
            return 1000 / (minutes * 60)
        if activity_type is ActivityType.SWIM:
            return 100 / (minutes * 60)
    except ValueError:
        pass
    return math.nan


def _first_present(row: pd.Series, *names: str) -> object:
    """Return the first of ``names`` that is a column of ``row``, else None."""
    for name in names:
        if name in row.index:
            return row[name]
    return None


@register_source("garmin")
class GarminCsvSource(ActivitySource):
    """Reads the Garmin Connect activity CSV export."""

    @override
    @classmethod
    def can_read(cls, path: Path) -> bool:
        return _REQUIRED.issubset(read_header(path))

    @override
    def read(self, path: Path, athlete_id: str) -> Iterator[Activity]:
        raw = pd.read_csv(path, dtype=str)
        for _, row in raw.iterrows():
            date = pd.to_datetime(row["Data"], errors="coerce")
            if pd.isna(date):
                continue
            atype = _TYPE_MAP.get(row["Typ aktywności"], ActivityType.WORKOUT)

            distance_km = to_float(row["Dystans"])
            if atype is ActivityType.SWIM and distance_km > MAX_SWIM_DISTANCE_KM:
                distance_km /= 1000

            temps = [
                to_float(row.get("Maksymalna temperatura")),
                to_float(row.get("Minimalna temperatura")),
            ]
            temps = [temp for temp in temps if not math.isnan(temp)]

            yield Activity(
                athlete_id=athlete_id,
                date=date.to_pydatetime(),
                type=atype,
                distance_m=distance_km * 1000,
                elapsed_s=parse_duration(_first_present(row, "Upłynęło czasu", "Czas")),
                moving_s=parse_duration(_first_present(row, "Czas ruchu", "Czas")),
                avg_speed_ms=speed_to_ms(
                    _first_present(row, "Średnie tempo", "Średnia prędkość"), atype
                ),
                max_speed_ms=speed_to_ms(
                    _first_present(row, "Najlepsze tempo", "Maksymalna prędkość"), atype
                ),
                avg_hr=to_float(row.get("Średnie tętno")),
                max_hr=to_float(row.get("Maksymalne tętno")),
                elevation_gain_m=to_float(row.get("Całkowity wznios")),
                elevation_loss_m=to_float(row.get("Całkowity spadek")),
                avg_temp_c=sum(temps) / len(temps) if temps else math.nan,
            )
