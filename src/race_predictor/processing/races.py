"""Race detection strategies (Strategy pattern).

``HeuristicRaceDetector`` ports ``identify_races`` from the thesis notebook
``personalized_model.ipynb``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from race_predictor.processing.pipeline import Step

RACE_DISTANCES_M = {"5k": 5000, "10k": 10000, "half": 21100, "marathon": 42195}


class RaceDetector(Protocol):
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """Boolean Series (aligned to ``df.index``): True where the row is a race."""
        ...


def distance_label(distance_m: float, tolerance: float = 0.1) -> str:
    for label, d in RACE_DISTANCES_M.items():
        if d * (1 - tolerance) <= distance_m <= d * (1 + tolerance):
            return label
    return "other"


def is_standard_distance(distance_m: pd.Series, tolerance: float = 0.1) -> pd.Series:
    out = pd.Series(False, index=distance_m.index)
    for d in RACE_DISTANCES_M.values():
        out |= distance_m.between(d * (1 - tolerance), d * (1 + tolerance))
    return out


class StandardDistanceDetector:
    """Simple baseline: any run within tolerance of 5k/10k/half/marathon (>=3 km)."""

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance

    def detect(self, df: pd.DataFrame) -> pd.Series:
        return (df["type"] == "Run") & is_standard_distance(df["distance_m"], self.tolerance)


class HeuristicRaceDetector:
    """Score-based heuristic: high HR, ultra distance, speed spike vs. 365-day average,
    triathlon run, standard distance; plus min 3 km and low stopped-time ratio."""

    def __init__(
        self,
        hr_threshold=0.86,
        days_window=365,
        min_distance_m=3000.0,
        max_time_ratio=1.1,
        min_score=2,
    ):
        self.hr_threshold = hr_threshold
        self.days_window = days_window
        self.min_distance_m = min_distance_m
        self.max_time_ratio = max_time_ratio
        self.min_score = min_score

    def _rolling_speed(self, df: pd.DataFrame) -> pd.Series:
        out = pd.Series(np.nan, index=df.index)
        runs = df[df["type"] == "Run"]
        for _, sub in runs.groupby("athlete"):
            sub = sub.sort_values("date")
            out.loc[sub.index] = (
                sub.set_index("date")["avg_speed_ms"]
                .rolling(f"{self.days_window}D", min_periods=1)
                .mean()
                .to_numpy()
            )
        return out

    def detect(self, df: pd.DataFrame) -> pd.Series:
        is_run = df["type"] == "Run"
        cond_hr = df["avg_hr"] > df["hr_max"] * self.hr_threshold
        cond_dist = df["distance_m"] > 40000
        cond_speed = df["avg_speed_ms"] > 1.4 * self._rolling_speed(df)
        tri = df.get("is_triathlon", pd.Series(0, index=df.index)).astype(bool)
        ratio = df["elapsed_s"] / df["moving_s"].replace(0, np.nan)
        score = (
            2 * cond_hr.astype(int)
            + 2 * cond_dist.astype(int)
            + 2 * cond_speed.astype(int)
            + tri.astype(int)
            + is_standard_distance(df["distance_m"]).astype(int)
        )
        return (
            is_run
            & (df["distance_m"] >= self.min_distance_m)
            & (score >= self.min_score)
            & (ratio <= self.max_time_ratio)
        )


class DetectRaces(Step):
    def __init__(self, detector: RaceDetector | None = None):
        self.detector = detector or HeuristicRaceDetector()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_race"] = self.detector.detect(df).fillna(False).astype(bool)
        return df
