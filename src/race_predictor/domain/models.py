"""Domain model: activities, athletes and races."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ActivityType(StrEnum):
    RUN = "Run"
    RIDE = "Ride"
    SWIM = "Swim"
    WEIGHT_TRAINING = "Weight Training"
    WORKOUT = "Workout"


NAN = math.nan


@dataclass(frozen=True, slots=True)
class Activity:
    """A single recorded activity in canonical units (metres, seconds, m/s, bpm)."""

    athlete_id: str
    date: datetime
    type: ActivityType
    distance_m: float = NAN
    elapsed_s: float = NAN
    moving_s: float = NAN
    avg_speed_ms: float = NAN
    max_speed_ms: float = NAN
    avg_hr: float = NAN
    max_hr: float = NAN
    elevation_gain_m: float = NAN
    elevation_loss_m: float = NAN
    avg_temp_c: float = NAN


@dataclass(frozen=True, slots=True)
class Race:
    """A run detected as a race."""

    activity: Activity
    label: str  # "5k", "10k", "half", "marathon" or "other"


@dataclass(slots=True)
class Athlete:
    id: str
    source: str
    activities: list[Activity] = field(default_factory=list)

    @property
    def hr_max(self) -> float:
        """Estimated HRmax: the highest max heart rate ever recorded."""
        values = [a.max_hr for a in self.activities if not math.isnan(a.max_hr)]
        return max(values) if values else NAN

    def runs(self) -> list[Activity]:
        return sorted(
            (a for a in self.activities if a.type is ActivityType.RUN), key=lambda a: a.date
        )
