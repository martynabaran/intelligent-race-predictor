"""Builders shared by the training-load tests.

Kept out of a ``test_*`` module and imported by plain module name (pytest puts this directory on
``sys.path``), so tests also collect under a bare ``pytest``, which is what CI runs.
"""

import pandas as pd


def activities(rows, **defaults):
    base = {
        "athlete": "a",
        "type": "Run",
        "moving_s": 3600.0,
        "elapsed_s": 3600.0,
        "avg_hr": 150.0,
        "max_hr": 190.0,
        "distance_m": 10_000.0,
        "avg_speed_ms": 3.0,
        "max_speed_ms": 4.0,
        "elevation_gain_m": 10.0,
        "elevation_loss_m": 10.0,
        "avg_temp_c": 15.0,
        **defaults,
    }
    df = pd.DataFrame([{**base, **row} for row in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def history(n_days=120, **extra):
    return activities([{"date": d, **extra} for d in pd.date_range("2024-01-01", periods=n_days)])


def training_frame():
    rows = []
    for athlete, speed in (("a", 3.0), ("b", 3.5), ("c", 4.0)):
        for i in range(60):
            rows.append(
                {
                    "athlete": athlete,
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=4 * i),
                    "avg_speed_ms": speed,
                    "is_race": False,
                    "is_triathlon": 0,
                    "hr_max": 190.0,
                }
            )
        for j in range(3):
            rows.append(
                {
                    "athlete": athlete,
                    "date": pd.Timestamp("2023-10-01") + pd.Timedelta(days=21 * j),
                    "avg_speed_ms": speed * 1.1,
                    "moving_s": 10_000 / (speed * 1.1),
                    "is_race": True,
                    "is_triathlon": 0,
                    "hr_max": 190.0,
                }
            )
    return activities(rows)
