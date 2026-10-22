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


class RaceMode(StrEnum):
    """Kind of race being predicted; models are chosen per mode (see ``models.router``).

    Only road running races for now. Runs after a ride are not predicted as road races: the
    manual review of the 39 weakest candidates (``docs/REVIEW_RACES.md``) found 27 brick training
    sessions and 12 real races, so the detector cannot tell them apart, and the real ones are
    triathlon run legs - slower than road races and too few for a model of their own. Trail races
    are not a separate mode either: only 9 races have >200 m of climbing.
    """

    ROAD = "road"


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
