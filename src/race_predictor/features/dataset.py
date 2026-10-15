"""Build one training example per detected race.

Port of ``build_sequences_and_statics_split`` from ``notebooks/personalized_model.ipynb``, but
without the split baked in (splitting lives in :mod:`race_predictor.evaluation.splits`).
Target: race pace in seconds per km.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    seqs: list[np.ndarray]  # (n_runs, len(SEQ_FEATURES)) each, oldest first
    static: pd.DataFrame  # STATIC_NUMERIC + STATIC_BINARY + AGGREGATES + riegel reference
    y: np.ndarray  # race pace [s/km]
    meta: pd.DataFrame  # athlete, race_date, race_distance_m

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx) -> RaceDataset:
        idx = np.asarray(idx)
        return RaceDataset(
            [self.seqs[i] for i in idx],
            self.static.iloc[idx].reset_index(drop=True),
            self.y[idx],
            self.meta.iloc[idx].reset_index(drop=True),
        )

    def shared_seqs(self) -> list[np.ndarray]:
        idx = [SEQ_FEATURES.index(f) for f in SHARED_SEQ_FEATURES]
        return [s[:, idx] for s in self.seqs]

    @property
    def tabular_columns(self) -> list[str]:
        return STATIC_NUMERIC + STATIC_BINARY + AGGREGATES

    @property
    def race_time_s(self) -> np.ndarray:
        return self.y * self.meta["race_distance_m"].to_numpy() / 1000.0


def _sum(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].sum(min_count=1) or 0.0)


def _aggregates(window: pd.DataFrame, race_date: pd.Timestamp, days_window: int) -> dict:
    pace = 1000.0 / window["avg_speed_ms"].replace(0, np.nan)
    km = window["distance_m"] / 1000.0
    age = (race_date - window["date"]).dt.days
    long_enough = window["distance_m"] >= 3000
    return {
        "n_runs": float(len(window)),
        "total_km": float(km.sum()),
        "mean_pace_s_km": float(pace.mean()),
        "best_pace_s_km": float(pace[long_enough].min()) if long_enough.any() else np.nan,
        "longest_run_km": float(km.max()),
        "mean_hr": float(window["avg_hr"].mean()),
        "km_last_28d": float(km[age <= 28].sum()),
        "km_last_90d": float(km[age <= 90].sum()),
        "weekly_km": float(km.sum() / (days_window / 7)),
    }


def _riegel_reference(window: pd.DataFrame) -> tuple[float, float]:
    """Fastest-pace run of >=5 km in the window -> (distance_m, time_s); NaNs if none."""
    cand = window[(window["distance_m"] >= 5000) & (window["avg_speed_ms"] > 0)]
    if cand.empty:
        return np.nan, np.nan
    best = cand.loc[cand["avg_speed_ms"].idxmax()]
    return float(best["distance_m"]), float(best["distance_m"] / best["avg_speed_ms"])


def build_race_dataset(df: pd.DataFrame, days_window: int = 365, min_runs: int = 3) -> RaceDataset:
    """``df`` must come from :func:`race_predictor.processing.default_pipeline`."""
    df = df.sort_values(["athlete", "date"]).reset_index(drop=True)
    seqs, statics, ys, metas = [], [], [], []

    for athlete, sub in df.groupby("athlete"):
        runs = sub[sub["type"] == "Run"]
        years_exp = (runs["date"].max() - runs["date"].min()).days / 365.0 if len(runs) > 1 else 0.0

        for _, race in runs[runs["is_race"]].iterrows():
            race_date = race["date"]
            start = race_date - pd.Timedelta(days=days_window)
            window = runs[(runs["date"] >= start) & (runs["date"] < race_date)].copy()
            if len(window) < min_runs:
                continue
            moving = race["moving_s"] if pd.notna(race["moving_s"]) else race["elapsed_s"]
            if pd.isna(moving) or race["distance_m"] <= 0:
                continue

            window["days_to_race"] = (race_date - window["date"]).dt.days
            others = sub[
                (sub["type"] != "Run") & (sub["date"] >= start) & (sub["date"] <= race_date)
            ]
            swim, bike, gym = (
                others[others["type"] == t] for t in ("Swim", "Ride", "Weight Training")
            )
            ref_d, ref_t = _riegel_reference(window)

            statics.append(
                {
                    "years_of_experience": years_exp,
                    "swim_t_total": _sum(swim, "elapsed_s"),
                    "swim_dist_total": _sum(swim, "distance_m"),
                    "swim_sessions": float(len(swim)),
                    "bike_t_total": _sum(bike, "elapsed_s"),
                    "bike_dist_total": _sum(bike, "distance_m"),
                    "bike_sessions": float(len(bike)),
                    "gym_t_total": _sum(gym, "elapsed_s"),
                    "gym_sessions": float(len(gym)),
                    "race_distance_m": float(race["distance_m"]),
                    "race_elevation_gain_m": float(race["elevation_gain_m"]),
                    "is_triathlon": float(race["is_triathlon"]),
                    **_aggregates(window, race_date, days_window),
                    "riegel_ref_distance_m": ref_d,
                    "riegel_ref_time_s": ref_t,
                }
            )
            seqs.append(window[SEQ_FEATURES].to_numpy(dtype=np.float32))
            ys.append(float(moving) / (race["distance_m"] / 1000.0))
            metas.append(
                {
                    "athlete": athlete,
                    "race_date": race_date,
                    "race_distance_m": float(race["distance_m"]),
                }
            )

    return RaceDataset(
        seqs, pd.DataFrame(statics), np.array(ys, dtype=np.float64), pd.DataFrame(metas)
    )
