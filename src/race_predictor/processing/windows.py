"""Training-window generators (leak-free by construction)."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from race_predictor.domain import ActivityType


def training_windows(
    df: pd.DataFrame, days_window: int = 365, min_runs: int = 3
) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
    """Yield ``(race_row, preceding_runs)`` for every detected race.

    ``preceding_runs`` contains only runs from ``[race_date - days_window, race_date)`` of the
    same athlete – strictly before the race, so no information from the race itself leaks in.
    Races with fewer than ``min_runs`` preceding runs are skipped.
    """
    runs = df[df["type"] == ActivityType.RUN].sort_values("date")
    for _, athlete_runs in runs.groupby("athlete"):
        for _, race in athlete_runs[athlete_runs["is_race"]].iterrows():
            start = race["date"] - pd.Timedelta(days=days_window)
            window = athlete_runs[
                (athlete_runs["date"] >= start) & (athlete_runs["date"] < race["date"])
            ]
            if len(window) >= min_runs:
                yield race, window
