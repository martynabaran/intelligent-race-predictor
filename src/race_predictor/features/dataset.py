"""Build one training example per detected race.

Port of ``build_sequences_and_statics_split`` from ``notebooks/personalized_model.ipynb``, but
without the split baked in (splitting lives in :mod:`race_predictor.evaluation.splits`).
Target: race pace in seconds per km.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from race_predictor.domain import ActivityType, RaceMode
from race_predictor.exceptions import DataValidationError
from race_predictor.training_load import LOAD_FEATURES, load_features, load_state

SEQ_FEATURES = [
    "days_to_race",
    "distance_m",
    "elapsed_s",
    "moving_s",
    "avg_speed_ms",
    "max_speed_ms",
    "avg_hr",
    "max_hr",
    "elevation_gain_m",
    "elevation_loss_m",
    "avg_temp_c",
]
# Channels available in both the personal exports and the Kaggle marathon data (used for transfer).
SHARED_SEQ_FEATURES = ["days_to_race", "distance_m", "moving_s", "avg_speed_ms"]
STATIC_NUMERIC = [
    "years_of_experience",
    "swim_t_total",
    "swim_dist_total",
    "swim_sessions",
    "bike_t_total",
    "bike_dist_total",
    "bike_sessions",
    "gym_t_total",
    "gym_sessions",
    "race_distance_m",
    "race_elevation_gain_m",
]
STATIC_BINARY = ["is_triathlon"]
# Distance thresholds [m] for the training-window summaries below.
MIN_PACE_RUN_M = 3_000.0  # shorter runs do not count towards the best training pace
MIN_REFERENCE_RUN_M = 5_000.0  # the shortest run the Riegel formula projects from
# Lengths [days] of the recent-volume windows (the km_last_28d / km_last_90d columns).
SHORT_WINDOW_DAYS = 28
LONG_WINDOW_DAYS = 90
# Hand-crafted summaries of the 365-day window, for tabular models (and Riegel).
AGGREGATES = [
    "n_runs",
    "total_km",
    "mean_pace_s_km",
    "best_pace_s_km",
    "longest_run_km",
    "mean_hr",
    "km_last_28d",
    "km_last_90d",
    "weekly_km",
]


@dataclass
class RaceDataset:
    """Features and targets of a set of races (real or hypothetical), one row per race.

    Attributes:
        seqs: Run sequence before each race, each ``(n_runs, len(SEQ_FEATURES))``, oldest first.
        static: Per-race tabular features (``STATIC_NUMERIC``, ``STATIC_BINARY``, ``AGGREGATES``,
            the Riegel reference and the training-load columns).
        y: Race pace in seconds per km.
        meta: ``athlete``, ``race_date`` and ``race_distance_m`` of each race.
        include_load: Whether ``tabular_columns`` contains the training-load features.
    """

    seqs: list[np.ndarray]  # (n_runs, len(SEQ_FEATURES)) each, oldest first
    static: pd.DataFrame  # STATIC_NUMERIC + STATIC_BINARY + AGGREGATES + riegel reference
    y: np.ndarray  # race pace [s/km]
    meta: pd.DataFrame  # athlete, race_date, race_distance_m
    # Training-load columns (LOAD_FEATURES) are always in ``static``; models see them through
    # ``tabular_columns`` only when this is set, so earlier results stay comparable.
    include_load: bool = False

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: Sequence[int] | np.ndarray) -> RaceDataset:
        """Return the races at positions ``idx`` (a boolean mask or integer positions)."""
        indices = np.asarray(idx)
        # A mask becomes positions (iterating a mask would index the sequence list with True/False);
        # an empty list would otherwise be a float array and unusable as an index.
        positions = np.flatnonzero(indices) if indices.dtype == bool else indices.astype(np.intp)
        return RaceDataset(
            [self.seqs[i] for i in positions],
            self.static.iloc[positions].reset_index(drop=True),
            self.y[positions],
            self.meta.iloc[positions].reset_index(drop=True),
            self.include_load,
        )

    def with_load(self, *, enabled: bool = True) -> RaceDataset:
        """Same data; ``tabular_columns`` includes the training-load features iff ``enabled``."""
        return replace(self, include_load=enabled)

    def with_targets(self, y: np.ndarray) -> RaceDataset:
        """Same features and settings with different targets (e.g. blended for distillation)."""
        return replace(self, y=y)

    def shared_seqs(self) -> list[np.ndarray]:
        """Return the run sequences restricted to the channels shared with the Kaggle data."""
        channels = [SEQ_FEATURES.index(name) for name in SHARED_SEQ_FEATURES]
        return [seq[:, channels] for seq in self.seqs]

    @property
    def tabular_columns(self) -> list[str]:
        """Columns of ``static`` that tabular models use."""
        columns = STATIC_NUMERIC + STATIC_BINARY + AGGREGATES
        return columns + LOAD_FEATURES if self.include_load else columns

    @property
    def race_time_s(self) -> np.ndarray:
        """Race time in seconds, from the pace and race distance."""
        return self.y * self.meta["race_distance_m"].to_numpy() / 1000.0


def _sum(frame: pd.DataFrame, column: str) -> float:
    """Sum of ``column``; 0 when it is empty or all NaN."""
    return float(frame[column].sum(min_count=1) or 0.0)


def _aggregates(
    window: pd.DataFrame, race_date: pd.Timestamp, days_window: int
) -> dict[str, float]:
    """Hand-crafted summaries of the runs in ``window`` (see ``AGGREGATES``)."""
    pace = 1000.0 / window["avg_speed_ms"].replace(0, np.nan)
    km = window["distance_m"] / 1000.0
    age = (race_date - window["date"]).dt.days
    long_enough = window["distance_m"] >= MIN_PACE_RUN_M
    return {
        "n_runs": float(len(window)),
        "total_km": float(km.sum()),
        "mean_pace_s_km": float(pace.mean()),
        "best_pace_s_km": float(pace[long_enough].min()) if long_enough.any() else np.nan,
        "longest_run_km": float(km.max()),
        "mean_hr": float(window["avg_hr"].mean()),
        "km_last_28d": float(km[age <= SHORT_WINDOW_DAYS].sum()),
        "km_last_90d": float(km[age <= LONG_WINDOW_DAYS].sum()),
        "weekly_km": float(km.sum() / (days_window / 7)),
    }


def _riegel_reference(window: pd.DataFrame) -> tuple[float, float, float]:
    """Fastest-pace run of >=5 km in the window -> (distance_m, time_s, is_race); NaNs if none.

    ``is_race`` (1/0) says whether that run was itself a detected race - Riegel is meant to
    project a race result, not a training run. NaN when races are unknown (e.g. a user's export).
    """
    candidates = window[
        (window["distance_m"] >= MIN_REFERENCE_RUN_M) & (window["avg_speed_ms"] > 0)
    ]
    if candidates.empty:
        return np.nan, np.nan, np.nan
    best = candidates.iloc[int(np.argmax(candidates["avg_speed_ms"].to_numpy()))]
    is_race = float(bool(best["is_race"])) if "is_race" in best.index else np.nan
    return float(best["distance_m"]), float(best["distance_m"] / best["avg_speed_ms"]), is_race


def _example(
    sub: pd.DataFrame,
    runs: pd.DataFrame,
    years_exp: float,
    race_date: pd.Timestamp,
    distance_m: float,
    elevation_gain_m: float,
    is_triathlon: float,
    days_window: int,
    min_runs: int,
) -> tuple[dict[str, float], np.ndarray] | None:
    """Static features + run sequence for a (real or hypothetical) race on ``race_date``.

    Only runs strictly before ``race_date`` are used. ``None`` if there are too few of them.

    Args:
        sub: All activities of the athlete (any type); source of the cross-training totals.
        runs: The athlete's runs only; source of the sequence and the run aggregates.
        years_exp: Years between the athlete's first and last run.
        race_date: Date of the race.
        distance_m: Race distance in metres.
        elevation_gain_m: Race elevation gain in metres.
        is_triathlon: 1.0 for a triathlon, else 0.0.
        days_window: Length of the look-back window before the race, in days.
        min_runs: Minimum number of runs in the window.
    """
    start = race_date - pd.Timedelta(days=days_window)
    window = runs[(runs["date"] >= start) & (runs["date"] < race_date)].copy()
    if len(window) < min_runs:
        return None
    window["days_to_race"] = (race_date - window["date"]).dt.days
    others = sub[
        (sub["type"] != ActivityType.RUN) & (sub["date"] >= start) & (sub["date"] <= race_date)
    ]
    swim, bike, gym = (
        others[others["type"] == activity_type]
        for activity_type in (ActivityType.SWIM, ActivityType.RIDE, ActivityType.WEIGHT_TRAINING)
    )
    ref_d, ref_t, ref_is_race = _riegel_reference(window)
    static = {
        "years_of_experience": years_exp,
        "swim_t_total": _sum(swim, "elapsed_s"),
        "swim_dist_total": _sum(swim, "distance_m"),
        "swim_sessions": float(len(swim)),
        "bike_t_total": _sum(bike, "elapsed_s"),
        "bike_dist_total": _sum(bike, "distance_m"),
        "bike_sessions": float(len(bike)),
        "gym_t_total": _sum(gym, "elapsed_s"),
        "gym_sessions": float(len(gym)),
        "race_distance_m": float(distance_m),
        "race_elevation_gain_m": float(elevation_gain_m),
        "is_triathlon": float(is_triathlon),
        **_aggregates(window, race_date, days_window),
        "riegel_ref_distance_m": ref_d,
        "riegel_ref_time_s": ref_t,
        "riegel_ref_is_race": ref_is_race,
    }
    return static, window[SEQ_FEATURES].to_numpy(dtype=np.float32)


def _years_of_experience(runs: pd.DataFrame) -> float:
    """Years between the first and last run (0 with fewer than two runs)."""
    return (runs["date"].max() - runs["date"].min()).days / 365.0 if len(runs) > 1 else 0.0


def build_race_dataset(df: pd.DataFrame, days_window: int = 365, min_runs: int = 3) -> RaceDataset:
    """Build one example per detected race.

    ``df`` must come from ``cleaning_pipeline("train")`` (:mod:`race_predictor.processing`).
    """
    df = df.sort_values(["athlete", "date"]).reset_index(drop=True)
    seqs: list[np.ndarray] = []
    statics: list[dict[str, float]] = []
    ys: list[float] = []
    metas: list[dict[str, object]] = []

    for athlete, activities in df.groupby("athlete"):
        runs = activities[activities["type"] == ActivityType.RUN]
        years_exp = _years_of_experience(runs)
        state = load_state(activities)  # causal: a race only sees the days before it (state_at)

        for _, race in runs[runs["is_race"]].iterrows():
            moving = race["moving_s"] if pd.notna(race["moving_s"]) else race["elapsed_s"]
            if pd.isna(moving) or race["distance_m"] <= 0:
                continue
            example = _example(
                activities,
                runs,
                years_exp,
                race["date"],
                race["distance_m"],
                race["elevation_gain_m"],
                race["is_triathlon"],
                days_window,
                min_runs,
            )
            if example is None:
                continue
            statics.append({**example[0], **load_features(state, race["date"])})
            seqs.append(example[1])
            ys.append(float(moving) / (race["distance_m"] / 1000.0))
            metas.append(
                {
                    "athlete": athlete,
                    "race_date": race["date"],
                    "race_distance_m": float(race["distance_m"]),
                }
            )

    if not statics:
        raise DataValidationError(
            f"no race has {min_runs} or more earlier runs in its {days_window}-day window"
        )
    return RaceDataset(
        seqs, pd.DataFrame(statics), np.array(ys, dtype=np.float64), pd.DataFrame(metas)
    )


def build_query_dataset(
    df: pd.DataFrame,
    distances_m: list[float],
    race_date: pd.Timestamp | None = None,
    elevation_gain_m: float = 0.0,
    days_window: int = 365,
    min_runs: int = 3,
    mode: RaceMode = RaceMode.ROAD,
) -> RaceDataset | None:
    """Features for *hypothetical* races of one athlete (``df`` = that athlete's activities).

    Default ``race_date`` is the day after the last activity ("a race today"). The target ``y`` is
    a placeholder (zeros). Returns ``None`` when the history before ``race_date`` is too short.
    ``mode`` is the kind of race (only road races for now, so ``is_triathlon`` is 0).
    """
    is_tri = 0.0 if RaceMode(mode) is RaceMode.ROAD else 1.0
    df = df.sort_values("date")
    race_date = race_date if race_date is not None else df["date"].max() + pd.Timedelta(days=1)
    runs = df[df["type"] == ActivityType.RUN]
    years_exp = _years_of_experience(runs[runs["date"] < race_date])
    before = df[df["date"] < race_date]  # the load state may only see the past
    if before.empty:
        return None
    state = load_state(before, end=race_date - pd.Timedelta(days=1))
    seqs: list[np.ndarray] = []
    statics: list[dict[str, float]] = []
    metas: list[dict[str, object]] = []
    for distance_m in distances_m:
        example = _example(
            df,
            runs,
            years_exp,
            race_date,
            distance_m,
            elevation_gain_m,
            is_tri,
            days_window,
            min_runs,
        )
        if example is None:
            return None
        statics.append({**example[0], **load_features(state, race_date)})
        seqs.append(example[1])
        metas.append(
            {
                "athlete": str(df["athlete"].iloc[0]),
                "race_date": race_date,
                "race_distance_m": distance_m,
            }
        )
    return RaceDataset(seqs, pd.DataFrame(statics), np.zeros(len(seqs)), pd.DataFrame(metas))
