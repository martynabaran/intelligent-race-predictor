"""High-level API used by the CLI and the Streamlit app.

``RacePredictor`` wraps the model chosen by the evaluation - a :class:`ModeRouter` (Riegel for road
races, ridge regression for the run leg of a triathlon, see ``docs/RESULTS_MODES.md``) - and adds
empirical prediction intervals per race mode from out-of-fold errors on unseen athletes, a Riegel
comparison, an exact explanation and a form-over-time curve.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class Prediction:
    """One predicted race - the contract between the prediction service and the UIs."""

    distance: str  # label, e.g. "10 km"
    distance_m: float
    time_s: float
    low_s: float  # empirical 10 % quantile of the error on unseen athletes (for ``mode``)
    high_s: float  # empirical 90 % quantile
    riegel_s: float  # plain Riegel projection, for comparison
    mode: RaceMode
    model: str  # name of the model used for ``mode``


PREDICTION_COLUMNS = [f.name for f in Prediction.__dataclass_fields__.values()]


def format_time(seconds: float) -> str:
    """Format seconds as ``h:mm:ss``; ``-`` for a missing or non-finite value."""
    if seconds is None or not np.isfinite(seconds):
        return "-"
    s = round(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def prepare_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Clean one athlete's canonical activities for prediction.

    Uses the same shared steps as training, with heart rate optional; see
    :mod:`race_predictor.processing.cleaning`.
    """
    return cleaning_pipeline("predict").run(df.assign(athlete=df["athlete"].astype(str)))


class RacePredictor:
    """Race-time predictor with empirical prediction intervals, explanations and a form curve.

    Args:
        model_factory: Zero-argument callable returning a fresh, unfitted model.
        interval: Lower and upper quantile of the out-of-fold error ratio used as the range.
        n_splits: Number of athlete-grouped folds for the out-of-fold errors.
        min_mode_races: Minimum races of a mode needed for its own interval; otherwise the
            pooled interval is used.
    """

    def __init__(
        self,
        model_factory: Callable[[], RaceModel] = ModeRouter,
        interval: tuple[float, float] = (0.1, 0.9),
        n_splits: int = 5,
        min_mode_races: int = 15,
    ) -> None:
        self.model_factory = model_factory
        self.interval = interval
        self.n_splits = n_splits
        self.min_mode_races = min_mode_races

    # ----- training -------------------------------------------------------------------------
    def fit(self, ds: RaceDataset) -> RacePredictor:
        """Fit the model on ``ds`` and estimate the per-mode intervals from out-of-fold errors."""
        self.model_: RaceModel = self.model_factory().fit(ds)
        fold_ratios: list[np.ndarray] = []
        fold_modes: list[np.ndarray] = []
        for tr, te in grouped_splits(ds, self.n_splits):
            test = ds.subset(te)
            fold_ratios.append(test.y / self.model_factory().fit(ds.subset(tr)).predict(test))
            fold_modes.append(race_modes(test))
        ratios = np.clip(np.concatenate(fold_ratios), 0.5, 2.0)
        modes = np.concatenate(fold_modes)
        # Interval per mode (errors differ a lot); pooled when a mode has too few races.
        pooled = tuple(np.quantile(ratios, self.interval))
        self.ratios_ = {
            mode: tuple(np.quantile(ratios[modes == mode], self.interval))
            if np.count_nonzero(modes == mode) >= self.min_mode_races
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
    def from_data_dir(cls, data_dir: str | Path = "data", **kwargs: Any) -> RacePredictor:
        """Load the bundled race dataset from ``data_dir`` and fit a predictor on it."""
        from race_predictor.evaluation.runner import load_dataset

        return cls(**kwargs).fit(load_dataset(Path(data_dir)))

    # ----- inference ------------------------------------------------------------------------
    def _query(
        self,
        activities: pd.DataFrame,
        distances_m: Iterable[float],
        race_date: pd.Timestamp | None,
        elevation_gain_m: float,
        mode: RaceMode = RaceMode.ROAD,
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

    def predictions(
        self,
        activities: pd.DataFrame,
        distances_m: dict[str, float] | None = None,
        race_date: pd.Timestamp | None = None,
        elevation_gain_m: float = 0.0,
        mode: RaceMode = RaceMode.ROAD,
    ) -> list[Prediction]:
        """Predicted time with an interval and the Riegel estimate, one per distance."""
        distances_m = distances_m or DISTANCES_M
        mode = RaceMode(mode)
        ds = self._query(activities, distances_m.values(), race_date, elevation_gain_m, mode)
        low, high = self.ratios_[mode]
        d = ds.meta["race_distance_m"].to_numpy()
        time = self._pace(ds) * d / 1000
        s = ds.static
        riegel = riegel_time(
            s["riegel_ref_time_s"].to_numpy(), s["riegel_ref_distance_m"].to_numpy(), d
        )
        model = self.model_name(mode)
        return [
            Prediction(
                label, float(dm), float(t), float(t * low), float(t * high), float(r), mode, model
            )
            for label, dm, t, r in zip(distances_m, d, time, riegel, strict=True)
        ]

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """:meth:`predictions` as a DataFrame with columns ``PREDICTION_COLUMNS``."""
        rows = [asdict(p) for p in self.predictions(*args, **kwargs)]
        return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)

    def explain(
        self,
        activities: pd.DataFrame,
        distance_m: float,
        race_date: pd.Timestamp | None = None,
        elevation_gain_m: float = 0.0,
        mode: RaceMode = RaceMode.ROAD,
    ) -> pd.Series:
        """Exact contributions to the predicted pace (s/km; + = slower) for ``mode``.

        They sum to the prediction. Ridge: per feature plus ``(baseline)``; Riegel: the
        reference run's pace plus the distance adjustment.
        """
        ds = self._query(activities, [distance_m], race_date, elevation_gain_m, mode)
        contrib = self.model_.contributions(ds).iloc[0]
        return contrib[contrib != 0].sort_values(key=lambda c: -c.abs())

    def reference_run(
        self, activities: pd.DataFrame, race_date: pd.Timestamp | None = None
    ) -> dict[str, float]:
        """The fastest >=5 km run of the last year that the Riegel formula projects from."""
        s = self._query(activities, [10000.0], race_date, 0.0).static.iloc[0]
        d, t = float(s["riegel_ref_distance_m"]), float(s["riegel_ref_time_s"])
        return {
            "distance_m": d,
            "time_s": t,
            "pace_s_km": t / (d / 1000) if d else np.nan,
            "is_race": bool(s["riegel_ref_is_race"] == 1),
        }

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
