"""High-level API used by the CLI and the Streamlit app.

``RacePredictor`` wraps the best robust model from the evaluation (ridge regression on tabular
training features), adds empirical prediction intervals from out-of-fold errors on unseen
athletes, a Riegel comparison, an exact linear explanation and a form-over-time curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.evaluation.splits import grouped_splits
from race_predictor.exceptions import DataValidationError
from race_predictor.features import RaceDataset, build_query_dataset
from race_predictor.models import RidgeBaseline
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import riegel_time
from race_predictor.processing import ImputeMissing, Pipeline, RemoveAnomalies, ValidateHeartRate

DISTANCES_M = {"5 km": 5000.0, "10 km": 10000.0, "Half marathon": 21097.5, "Marathon": 42195.0}
PACE_BOUNDS_S_KM = (150.0, 900.0)  # predictions are clipped to a plausible 2:30-15:00 /km


def format_time(seconds: float) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def prepare_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one athlete's canonical activities (HR optional, unlike the thesis pipeline)."""
    cleaned = Pipeline([RemoveAnomalies(), ValidateHeartRate(keep_missing=True), ImputeMissing()])
    return cleaned.run(df.assign(max_hr=df["max_hr"], athlete=df["athlete"].astype(str)))


class RacePredictor:
    def __init__(
        self,
        model_factory=RidgeBaseline,
        interval: tuple[float, float] = (0.1, 0.9),
        n_splits: int = 5,
    ):
        self.model_factory = model_factory
        self.interval = interval
        self.n_splits = n_splits

    # ----- training -------------------------------------------------------------------------
    def fit(self, ds: RaceDataset) -> RacePredictor:
        self.model_: RaceModel = self.model_factory().fit(ds)
        ratios = []
        for tr, te in grouped_splits(ds, self.n_splits):
            pred = self.model_factory().fit(ds.subset(tr)).predict(ds.subset(te))
            ratios.append(ds.subset(te).y / pred)
        ratios = np.clip(np.concatenate(ratios), 0.5, 2.0)
        self.ratio_low_, self.ratio_high_ = np.quantile(ratios, self.interval)
        self.n_train_races_ = len(ds)
        self.n_train_athletes_ = ds.meta["athlete"].nunique()
        return self

    @classmethod
    def from_data_dir(cls, data_dir: str | Path = "data", **kwargs) -> RacePredictor:
        from race_predictor.evaluation.runner import load_dataset

        return cls(**kwargs).fit(load_dataset(Path(data_dir)))

    # ----- inference ------------------------------------------------------------------------
    def _query(self, activities, distances_m, race_date, elevation_gain_m) -> RaceDataset:
        df = prepare_activities(activities)
        ds = build_query_dataset(df, list(distances_m), race_date, elevation_gain_m)
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
    ) -> pd.DataFrame:
        """Predicted time with an interval and the Riegel estimate, one row per distance."""
        distances_m = distances_m or DISTANCES_M
        ds = self._query(activities, distances_m.values(), race_date, elevation_gain_m)
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
                "low_s": time * self.ratio_low_,
                "high_s": time * self.ratio_high_,
                "riegel_s": riegel,
            }
        )

    def explain(
        self, activities: pd.DataFrame, distance_m: float, race_date=None, elevation_gain_m=0.0
    ) -> pd.Series:
        """Exact per-feature contribution (s/km of pace; + = slower) for a ridge model, relative
        to the average training race. Includes ``(baseline)`` = intercept + mean effect."""
        ds = self._query(activities, [distance_m], race_date, elevation_gain_m)
        pipe = self.model_.model_
        imputer, scaler, ridge = pipe[0], pipe[1], pipe[2]
        z = scaler.transform(imputer.transform(self.model_._x(ds)))[0]
        contrib = pd.Series(z * ridge.coef_, index=ds.tabular_columns)
        contrib["(baseline)"] = float(ridge.intercept_)
        return contrib.sort_values(key=lambda c: -c.abs())

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
