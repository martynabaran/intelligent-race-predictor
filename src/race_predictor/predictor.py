"""High-level API used by the CLI and the Streamlit app.

``RacePredictor`` wraps the model chosen by the evaluation - a :class:`ModeRouter` (Riegel for road
races, ridge regression for the run leg of a triathlon, see ``docs/RESULTS_MODES.md``) - and adds
empirical prediction intervals per race mode from out-of-fold errors on unseen athletes, a Riegel
comparison, an exact explanation and a form-over-time curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.domain import RaceMode
from race_predictor.evaluation.splits import grouped_splits
from race_predictor.exceptions import DataValidationError
from race_predictor.features import RaceDataset, build_query_dataset
from race_predictor.models import ModeRouter, race_modes
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import riegel_time
from race_predictor.processing import cleaning_pipeline

DISTANCES_M = {"5 km": 5000.0, "10 km": 10000.0, "Half marathon": 21097.5, "Marathon": 42195.0}
PACE_BOUNDS_S_KM = (150.0, 900.0)  # predictions are clipped to a plausible 2:30-15:00 /km


def format_time(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def prepare_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one athlete's canonical activities for prediction (same shared steps as training,
    heart rate optional; see :mod:`race_predictor.processing.cleaning`)."""
    return cleaning_pipeline("predict").run(df.assign(athlete=df["athlete"].astype(str)))


class RacePredictor:
    def __init__(
        self,
        model_factory=ModeRouter,
        interval: tuple[float, float] = (0.1, 0.9),
        n_splits: int = 5,
        min_mode_races: int = 15,
    ):
        self.model_factory = model_factory
        self.interval = interval
        self.n_splits = n_splits
        self.min_mode_races = min_mode_races

    # ----- training -------------------------------------------------------------------------
    def fit(self, ds: RaceDataset) -> RacePredictor:
        self.model_: RaceModel = self.model_factory().fit(ds)
        ratios, modes = [], []
        for tr, te in grouped_splits(ds, self.n_splits):
            test = ds.subset(te)
            ratios.append(test.y / self.model_factory().fit(ds.subset(tr)).predict(test))
            modes.append(race_modes(test))
        ratios, modes = np.clip(np.concatenate(ratios), 0.5, 2.0), np.concatenate(modes)
        # Interval per mode (errors differ a lot); pooled when a mode has too few races.
        pooled = tuple(np.quantile(ratios, self.interval))
        self.ratios_ = {
            mode: tuple(np.quantile(ratios[modes == mode], self.interval))
            if (modes == mode).sum() >= self.min_mode_races
            else pooled
            for mode in RaceMode
        }
        self.n_train_races_ = len(ds)
        self.n_train_athletes_ = ds.meta["athlete"].nunique()
        return self

    def model_name(self, mode: RaceMode = RaceMode.ROAD) -> str:
        """Name of the model used for ``mode``."""
        model = self.model_.model_for(mode) if hasattr(self.model_, "model_for") else self.model_
        return model.name

    @classmethod
    def from_data_dir(cls, data_dir: str | Path = "data", **kwargs) -> RacePredictor:
        from race_predictor.evaluation.runner import load_dataset

        return cls(**kwargs).fit(load_dataset(Path(data_dir)))

    # ----- inference ------------------------------------------------------------------------
    def _query(
        self, activities, distances_m, race_date, elevation_gain_m, mode=RaceMode.ROAD
    ) -> RaceDataset:
        df = prepare_activities(activities)
        ds = build_query_dataset(df, list(distances_m), race_date, elevation_gain_m, mode=mode)
        if ds is None:
            raise DataValidationError(
                "Need at least 3 runs in the 365 days before the race date to make a prediction."
            )
        return ds

    def _pace(self, ds: RaceDataset) -> np.ndarray:
        return np.clip(self.model_.predict(ds), *PACE_BOUNDS_S_KM)

    def predict(
        self,
        activities: pd.DataFrame,
        distances_m: dict[str, float] | None = None,
        race_date: pd.Timestamp | None = None,
        elevation_gain_m: float = 0.0,
        mode: RaceMode = RaceMode.ROAD,
    ) -> pd.DataFrame:
        """Predicted time with an interval and the Riegel estimate, one row per distance."""
        distances_m = distances_m or DISTANCES_M
        ds = self._query(activities, distances_m.values(), race_date, elevation_gain_m, mode)
        low, high = self.ratios_[RaceMode(mode)]
        pace = self._pace(ds)
        d = ds.meta["race_distance_m"].to_numpy()
        time = pace * d / 1000
        s = ds.static
        riegel = riegel_time(s["riegel_ref_time_s"], s["riegel_ref_distance_m"], d).to_numpy()
        return pd.DataFrame(
            {
                "distance": list(distances_m),
                "distance_m": d,
                "time_s": time,
                "low_s": time * low,
                "high_s": time * high,
                "riegel_s": riegel,
            }
        )

    def explain(
        self,
        activities: pd.DataFrame,
        distance_m: float,
        race_date=None,
        elevation_gain_m=0.0,
        mode: RaceMode = RaceMode.ROAD,
    ) -> pd.Series:
        """Exact contributions to the predicted pace (s/km; + = slower) from the model used for
        ``mode``; they sum to the prediction. Ridge: per feature plus ``(baseline)``; Riegel: the
        reference run's pace plus the distance adjustment."""
        ds = self._query(activities, [distance_m], race_date, elevation_gain_m, mode)
        contrib = self.model_.contributions(ds).iloc[0]
        return contrib[contrib != 0].sort_values(key=lambda c: -c.abs())

    def reference_run(self, activities: pd.DataFrame, race_date=None) -> dict[str, float]:
        """The fastest >=5 km run of the last year that the Riegel formula projects from."""
        s = self._query(activities, [10000.0], race_date, 0.0).static.iloc[0]
        d, t = float(s["riegel_ref_distance_m"]), float(s["riegel_ref_time_s"])
        return {"distance_m": d, "time_s": t, "pace_s_km": t / (d / 1000) if d else np.nan}

    def form_curve(
        self,
        activities: pd.DataFrame,
        distance_m: float = 10000.0,
        step_days: int = 14,
        warmup_days: int = 90,
    ) -> pd.DataFrame:
        """Predicted time for ``distance_m`` if a race had been run on each past date."""
        df = prepare_activities(activities)
        first, last = df["date"].min(), df["date"].max()
        rows = []
        for date in pd.date_range(
            first + pd.Timedelta(days=warmup_days), last, freq=f"{step_days}D"
        ):
            ds = build_query_dataset(df, [distance_m], date)
            if ds is not None:
                rows.append({"date": date, "time_s": float(self._pace(ds)[0] * distance_m / 1000)})
        return pd.DataFrame(rows, columns=["date", "time_s"])
