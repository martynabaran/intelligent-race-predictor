"""Race detection strategies (Strategy pattern).

``HeuristicRaceDetector`` ports ``identify_races`` from the thesis notebook
``personalized_model.ipynb``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from typing_extensions import override

from race_predictor.domain import MARATHON_M, ActivityType
from race_predictor.processing.pipeline import Step

RACE_DISTANCES_M = {"5k": 5000, "10k": 10000, "half": 21100, "marathon": MARATHON_M}
ULTRA_DISTANCE_M = 40_000.0  # runs longer than this score as a race on distance alone


class RaceDetector(Protocol):
    """Anything that can decide which rows of an activity frame are races."""

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Boolean Series (aligned to ``df.index``): True where the row is a race."""
        ...


def distance_label(distance_m: float, tolerance: float = 0.1) -> str:
    """Name the standard race distance within ``tolerance`` of ``distance_m``, else ``"other"``."""
    for label, standard_m in RACE_DISTANCES_M.items():
        if standard_m * (1 - tolerance) <= distance_m <= standard_m * (1 + tolerance):
            return label
    return "other"


def is_standard_distance(distance_m: pd.Series, tolerance: float = 0.1) -> pd.Series:
    """Flag distances within ``tolerance`` of any standard race distance."""
    flags = pd.Series(data=False, index=distance_m.index)
    for standard_m in RACE_DISTANCES_M.values():
        flags |= distance_m.between(standard_m * (1 - tolerance), standard_m * (1 + tolerance))
    return flags


class StandardDistanceDetector:
    """Simple baseline: any run within tolerance of 5k/10k/half/marathon (>=3 km)."""

    def __init__(self, tolerance: float = 0.1) -> None:
        self.tolerance = tolerance

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Flag runs of standard race distance."""
        return (df["type"] == ActivityType.RUN) & is_standard_distance(
            df["distance_m"], self.tolerance
        )


class HeuristicRaceDetector:
    """Score-based heuristic.

    A run scores points for high HR, ultra distance, a speed spike versus the 365-day average,
    being a triathlon run and having a standard distance; it must also cover at least
    ``min_distance_m`` and have a low stopped-time ratio.
    """

    def __init__(
        self,
        hr_threshold: float = 0.86,
        days_window: int = 365,
        min_distance_m: float = 3000.0,
        max_time_ratio: float = 1.1,
        min_score: int = 2,
    ) -> None:
        self.hr_threshold = hr_threshold
        self.days_window = days_window
        self.min_distance_m = min_distance_m
        self.max_time_ratio = max_time_ratio
        self.min_score = min_score

    def _rolling_speed(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df.index)
        runs = df[df["type"] == ActivityType.RUN]
        for _, athlete_runs in runs.groupby("athlete"):
            athlete_runs = athlete_runs.sort_values("date")
            out.loc[athlete_runs.index] = (
                athlete_runs.set_index("date")["avg_speed_ms"]
                .rolling(f"{self.days_window}D", min_periods=1)
                .mean()
                .to_numpy()
            )
        return out

    def reasons(self, df: pd.DataFrame) -> pd.DataFrame:
        """Each scoring condition per row, the score and the decision (for reviewing labels)."""
        out = pd.DataFrame(
            {
                "high_hr": df["avg_hr"] > df["hr_max"] * self.hr_threshold,
                "ultra": df["distance_m"] > ULTRA_DISTANCE_M,
                "speed_spike": df["avg_speed_ms"] > 1.4 * self._rolling_speed(df),
                "after_ride": df.get("is_triathlon", pd.Series(0, index=df.index)).astype(bool),
                "standard_distance": is_standard_distance(df["distance_m"]),
            },
            index=df.index,
        )
        out["score"] = (
            2 * (out["high_hr"].astype(int) + out["ultra"] + out["speed_spike"])
            + out["after_ride"]
            + out["standard_distance"]
        )
        ratio = df["elapsed_s"] / df["moving_s"].replace(0, np.nan)
        out["is_race"] = (
            (df["type"] == ActivityType.RUN)
            & (df["distance_m"] >= self.min_distance_m)
            & (out["score"] >= self.min_score)
            & (ratio <= self.max_time_ratio)
        )
        return out

    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Flag the rows whose score reaches ``min_score``."""
        return self.reasons(df)["is_race"]


class DetectRaces(Step):
    """Add the boolean ``is_race`` column using a :class:`RaceDetector`."""

    def __init__(self, detector: RaceDetector | None = None) -> None:
        self.detector = detector or HeuristicRaceDetector()

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_race"] = self.detector.detect(df).fillna(value=False).astype(bool)
        return df
