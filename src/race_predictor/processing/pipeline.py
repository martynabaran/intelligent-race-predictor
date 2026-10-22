"""Cleaning pipeline: a chain of small, independently testable steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd

from race_predictor.exceptions import DataValidationError

REQUIRED_COLUMNS = {"athlete", "date", "type", "distance_m", "avg_speed_ms", "avg_hr", "max_hr"}


class Step(ABC):
    """One transformation of the activity DataFrame."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Pipeline:
    def __init__(self, steps: Iterable[Step]):
        self.steps = list(steps)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing columns: {sorted(missing)}")
        for step in self.steps:
            df = step.apply(df)
        return df


class RemoveAnomalies(Step):
    """Drop runs that are physically implausible or too short (thesis: >8 m/s avg, >15 m/s max,
    <1 km)."""

    def __init__(self, max_avg_speed=8.0, max_peak_speed=15.0, min_distance_m=1000.0):
        self.max_avg_speed = max_avg_speed
        self.max_peak_speed = max_peak_speed
        self.min_distance_m = min_distance_m

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        bad = (df["type"] == "Run") & (
            (df["avg_speed_ms"] > self.max_avg_speed)
            | (df["max_speed_ms"] > self.max_peak_speed)
            | (df["distance_m"] < self.min_distance_m)
        )
        return df[~bad].reset_index(drop=True)


class ValidateHeartRate(Step):
    """Keep rows whose average heart rate is physiologically possible. Rows without HR are kept
    for imputation by later steps only if ``keep_missing`` is set."""

    def __init__(self, low=40.0, high=220.0, keep_missing=False):
        self.low, self.high, self.keep_missing = low, high, keep_missing

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ok = df["avg_hr"].between(self.low, self.high)
        if self.keep_missing:
            ok |= df["avg_hr"].isna()
        return df[ok].reset_index(drop=True)


class ValidateMaxHeartRate(Step):
    """Blank out max heart rate readings above ``high`` (sensor spikes).

    ``EstimateHRMax`` takes each athlete's highest reading, so a single spike (up to 253 bpm in the
    thesis data) inflated HRmax and with it the heart-rate threshold used for race detection.
    """

    def __init__(self, high=220.0):
        self.high = high

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["max_hr"] = df["max_hr"].where(~(df["max_hr"] > self.high))
        return df


class RemoveImplausiblyFastRuns(Step):
    """Drop runs faster than ``max_rel_speed`` x the athlete's median run speed.

    In the thesis data genuine runs reach at most ~1.63x the athlete's median; the one run above
    1.7x (12 km at 2:38/km, average HR 131) is a ride logged as a run. As the athlete's "best"
    run it set ``best_pace_s_km`` and the Riegel reference for every one of their races.
    """

    def __init__(self, max_rel_speed=1.7):
        self.max_rel_speed = max_rel_speed

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        runs = df["type"] == "Run"
        median = df["avg_speed_ms"].where(runs).groupby(df["athlete"]).transform("median")
        bad = runs & (df["avg_speed_ms"] > self.max_rel_speed * median)
        return df[~bad].reset_index(drop=True)


class DropTriathlonRaces(Step):
    """Keep the run after a ride as training history, but never as a road race to predict.

    The detector flags runs after a ride at a standard distance as races but cannot tell brick
    training from triathlon run legs: of the 39 weakest cases the manual review found 27 training
    sessions and 12 races (``docs/REVIEW_RACES.md``). Run legs are slower than road races of the
    same distance and there are too few for a model of their own, so they are left out of the
    races to predict.
    """

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_race"] = df["is_race"] & (df["is_triathlon"] != 1)
        return df


class DropAthletesWithoutHR(Step):
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        has_hr = df.groupby("athlete")[["avg_hr", "max_hr"]].apply(lambda g: g.notna().any().any())
        return df[df["athlete"].isin(has_hr[has_hr].index)].reset_index(drop=True)


class ImputeMissing(Step):
    """Fill numeric NaNs with the athlete+type median, falling back to the type median."""

    def __init__(self, columns: Iterable[str] | None = None):
        self.columns = list(columns) if columns else None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = self.columns or [
            c for c in df.select_dtypes("number").columns if c not in {"hr_max"}
        ]
        for col in cols:
            by_athlete = df.groupby(["athlete", "type"])[col].transform("median")
            by_type = df.groupby("type")[col].transform("median")
            df[col] = df[col].fillna(by_athlete).fillna(by_type)
        return df


class EstimateHRMax(Step):
    """Per-athlete HRmax = highest max heart rate observed (adds ``hr_max`` column)."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hr_max"] = df.groupby("athlete")["max_hr"].transform("max")
        return df


class MarkTriathlonRuns(Step):
    """Flag a run that follows a ride within ``max_gap_hours`` (brick / triathlon run)."""

    def __init__(self, max_gap_hours: float = 6.0):
        self.max_gap_hours = max_gap_hours

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["athlete", "date"]).reset_index(drop=True)
        prev_type = df.groupby("athlete")["type"].shift()
        gap_h = (df["date"] - df.groupby("athlete")["date"].shift()).dt.total_seconds() / 3600
        df["is_triathlon"] = (
            (df["type"] == "Run") & (prev_type == "Ride") & (gap_h <= self.max_gap_hours)
        ).astype(int)
        return df
