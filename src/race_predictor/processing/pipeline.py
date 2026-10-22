"""Cleaning pipeline: a chain of small, independently testable steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd
from typing_extensions import override

from race_predictor.domain import ActivityType
from race_predictor.exceptions import DataValidationError

REQUIRED_COLUMNS = {"athlete", "date", "type", "distance_m", "avg_speed_ms", "avg_hr", "max_hr"}


class Step(ABC):
    """One transformation of the activity DataFrame."""

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed frame; the input must not be modified in place."""

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Pipeline:
    """An ordered chain of :class:`Step` objects applied one after another."""

    def __init__(self, steps: Iterable[Step]):
        self.steps = list(steps)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the required columns, then pass ``df`` through every step.

        Raises:
            DataValidationError: If a required column is missing.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"Missing columns: {sorted(missing)}")
        for step in self.steps:
            df = step.apply(df)
        return df


class RemoveAnomalies(Step):
    """Drop runs that are physically implausible or too short.

    The thesis used more than 8 m/s average, more than 15 m/s peak and less than 1 km.
    """

    def __init__(
        self,
        max_avg_speed: float = 8.0,
        max_peak_speed: float = 15.0,
        min_distance_m: float = 1000.0,
    ) -> None:
        self.max_avg_speed = max_avg_speed
        self.max_peak_speed = max_peak_speed
        self.min_distance_m = min_distance_m

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        bad = (df["type"] == ActivityType.RUN) & (
            (df["avg_speed_ms"] > self.max_avg_speed)
            | (df["max_speed_ms"] > self.max_peak_speed)
            | (df["distance_m"] < self.min_distance_m)
        )
        return df[~bad].reset_index(drop=True)


class ValidateHeartRate(Step):
    """Keep rows whose average heart rate is physiologically possible.

    Rows without HR are kept for imputation by later steps only if ``keep_missing`` is set.
    """

    def __init__(
        self, low: float = 40.0, high: float = 220.0, *, keep_missing: bool = False
    ) -> None:
        self.low, self.high, self.keep_missing = low, high, keep_missing

    @override
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

    def __init__(self, high: float = 220.0) -> None:
        self.high = high

    @override
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

    def __init__(self, max_rel_speed: float = 1.7) -> None:
        self.max_rel_speed = max_rel_speed

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        runs = df["type"] == ActivityType.RUN
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

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["is_race"] = df["is_race"] & (df["is_triathlon"] != 1)
        return df


class DropAthletesWithoutHR(Step):
    """Drop every athlete who has no heart-rate reading at all."""

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        has_hr = df[["avg_hr", "max_hr"]].notna().any(axis=1).groupby(df["athlete"]).any()
        return df[df["athlete"].isin(has_hr[has_hr].index)].reset_index(drop=True)


class ImputeMissing(Step):
    """Fill numeric NaNs with the athlete+type median, falling back to the type median."""

    def __init__(self, columns: Iterable[str] | None = None) -> None:
        self.columns = list(columns) if columns else None

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns = self.columns or [
            column for column in df.select_dtypes("number").columns if column != "hr_max"
        ]
        for column in columns:
            by_athlete = df.groupby(["athlete", "type"])[column].transform("median")
            by_type = df.groupby("type")[column].transform("median")
            df[column] = df[column].fillna(by_athlete).fillna(by_type)
        return df


class EstimateHRMax(Step):
    """Per-athlete HRmax = highest max heart rate observed (adds ``hr_max`` column)."""

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["hr_max"] = df.groupby("athlete")["max_hr"].transform("max")
        return df


class MarkTriathlonRuns(Step):
    """Flag a run that follows a ride within ``max_gap_hours`` (brick / triathlon run)."""

    def __init__(self, max_gap_hours: float = 6.0) -> None:
        self.max_gap_hours = max_gap_hours

    @override
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["athlete", "date"]).reset_index(drop=True)
        prev_type = df.groupby("athlete")["type"].shift()
        gap_hours = (df["date"] - df.groupby("athlete")["date"].shift()).dt.total_seconds() / 3600
        df["is_triathlon"] = (
            (df["type"] == ActivityType.RUN)
            & (prev_type == ActivityType.RIDE)
            & (gap_hours <= self.max_gap_hours)
        ).astype(int)
        return df
