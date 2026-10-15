"""Pretraining examples from the Kaggle marathon data: one per (athlete, marathon)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from race_predictor.features.dataset import SHARED_SEQ_FEATURES
from race_predictor.sources.kaggle import MARATHON_DATES

MARATHON_M = 42195.0


@dataclass
class PretrainData:
    seqs: list[np.ndarray]  # (n_runs, len(SHARED_SEQ_FEATURES)), oldest first
    y: np.ndarray  # marathon pace [s/km]
    athlete: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def _marathon_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that are an athlete's marathon (>42 km on a major's date, as in the notebook)."""
    date = df["datetime"].dt.strftime("%Y-%m-%d")
    hit = pd.Series(False, index=df.index)
    for name, d in MARATHON_DATES.items():
        hit |= (date == d) & df["major"].str.contains(name, regex=False, na=False)
    return df[hit & (df["distance"] > 42)]


def build_marathon_data(
    df: pd.DataFrame,
    days_window: int = 365,
    min_runs: int = 3,
    max_marathon_min: float = 420.0,
    min_marathon_min: float = 120.0,
) -> PretrainData:
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "distance", "duration"])
    targets = _marathon_rows(df)
    targets = targets[targets["duration"].between(min_marathon_min, max_marathon_min)]
    targets = targets.drop_duplicates(["athlete", "datetime"])

    acts = df[(df["distance"] > 0) & (df["duration"] > 0)].sort_values(["athlete", "datetime"])
    by_athlete = {a: g for a, g in acts.groupby("athlete")}

    seqs, ys, athletes = [], [], []
    for _, t in targets.iterrows():
        sub = by_athlete.get(t["athlete"])
        if sub is None:
            continue
        start = t["datetime"] - pd.Timedelta(days=days_window)
        window = sub[(sub["datetime"] >= start) & (sub["datetime"] < t["datetime"])]
        if len(window) < min_runs:
            continue
        dist_m = window["distance"].to_numpy(float) * 1000
        moving = window["duration"].to_numpy(float) * 60
        days = (t["datetime"] - window["datetime"]).dt.days.to_numpy(float)
        seq = np.column_stack([days, dist_m, moving, dist_m / moving]).astype(np.float32)
        assert seq.shape[1] == len(SHARED_SEQ_FEATURES)
        seqs.append(seq)
        ys.append(t["duration"] * 60 / (MARATHON_M / 1000))
        athletes.append(t["athlete"])
    return PretrainData(seqs, np.array(ys, float), np.array(athletes))
