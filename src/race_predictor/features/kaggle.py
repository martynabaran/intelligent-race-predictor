"""Pretraining examples from the Kaggle marathon data: one per (athlete, marathon)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from race_predictor.domain import MARATHON_M
from race_predictor.features.dataset import SHARED_SEQ_FEATURES
from race_predictor.sources.kaggle import MARATHON_DATES

MARATHON_MIN_KM = 42.0  # activities shorter than this on a major's date are not its marathon


@dataclass
class PretrainData:
    """Marathon pretraining examples: shared-channel run sequences, paces and athlete ids."""

    seqs: list[np.ndarray]  # (n_runs, len(SHARED_SEQ_FEATURES)), oldest first
    y: np.ndarray  # marathon pace [s/km]
    athlete: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _marathon_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are an athlete's marathon (>42 km on a major's date, as in the notebook)."""
    date = df["datetime"].dt.strftime("%Y-%m-%d")
    hit = pd.Series(data=False, index=df.index)
    for name, marathon_date in MARATHON_DATES.items():
        hit |= (date == marathon_date) & df["major"].str.contains(name, regex=False, na=False)
    return df[hit & (df["distance"] > MARATHON_MIN_KM)]


def build_marathon_data(
    df: pd.DataFrame,
    days_window: int = 365,
    min_runs: int = 3,
    max_marathon_min: float = 420.0,
    min_marathon_min: float = 120.0,
) -> PretrainData:
    """Build one pretraining example per athlete marathon in the Kaggle data.

    Args:
        df: Kaggle activity table (``athlete``, ``datetime``, ``distance`` km, ``duration`` min,
            ``major``).
        days_window: Length of the look-back window before the marathon, in days.
        min_runs: Minimum number of activities in the window.
        max_marathon_min: Slowest accepted marathon time, in minutes.
        min_marathon_min: Fastest accepted marathon time, in minutes.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "distance", "duration"])
    targets = _marathon_rows(df)
    targets = targets[targets["duration"].between(min_marathon_min, max_marathon_min)]
    targets = targets.drop_duplicates(["athlete", "datetime"])

    acts = df[(df["distance"] > 0) & (df["duration"] > 0)].sort_values(["athlete", "datetime"])
    by_athlete = dict(iter(acts.groupby("athlete")))

    seqs: list[np.ndarray] = []
    ys: list[float] = []
    athletes: list[object] = []
    for _, marathon in targets.iterrows():
        history = by_athlete.get(marathon["athlete"])
        if history is None:
            continue
        start = marathon["datetime"] - pd.Timedelta(days=days_window)
        window = history[
            (history["datetime"] >= start) & (history["datetime"] < marathon["datetime"])
        ]
        if len(window) < min_runs:
            continue
        dist_m = window["distance"].to_numpy(float) * 1000
        moving = window["duration"].to_numpy(float) * 60
        days = (marathon["datetime"] - window["datetime"]).dt.days.to_numpy(float)
        seq = np.column_stack([days, dist_m, moving, dist_m / moving]).astype(np.float32)
        assert seq.shape[1] == len(SHARED_SEQ_FEATURES)  # internal invariant
        seqs.append(seq)
        ys.append(marathon["duration"] * 60 / (MARATHON_M / 1000))
        athletes.append(marathon["athlete"])
    return PretrainData(seqs, np.array(ys, float), np.array(athletes))
