"""A race goal and the verdict on how realistic it is."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from race_predictor.exceptions import DataValidationError


class Verdict(StrEnum):
    """How realistic a goal is compared with the predicted time and its range."""

    REALISTIC = "realistic"  # at or slower than the prediction
    AMBITIOUS = "ambitious"  # faster than predicted, but inside the plausible range
    UNLIKELY = "unlikely"  # faster than the fast end of the plausible range


@dataclass(frozen=True, slots=True)
class Goal:
    """A target finish time for one race."""

    distance_m: float
    time_s: float
    race_date: date

    def __post_init__(self) -> None:
        if self.distance_m <= 0:
            raise DataValidationError(f"goal distance must be positive, got {self.distance_m}")
        if self.time_s <= 0:
            raise DataValidationError(f"goal time must be positive, got {self.time_s}")

    @property
    def pace_s_per_km(self) -> float:
        """Average pace needed to reach the goal, in seconds per kilometre."""
        return self.time_s / (self.distance_m / 1000)
