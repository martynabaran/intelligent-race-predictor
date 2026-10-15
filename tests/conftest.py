from datetime import datetime, timedelta

import pandas as pd
import pytest


def make_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "athlete": "a",
        "type": "Run",
        "distance_m": 10000.0,
        "elapsed_s": 3000.0,
        "moving_s": 3000.0,
        "avg_speed_ms": 3.3,
        "max_speed_ms": 4.5,
        "avg_hr": 150.0,
        "max_hr": 180.0,
        "elevation_gain_m": 10.0,
        "elevation_loss_m": 10.0,
        "avg_temp_c": 15.0,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def frame():
    return make_frame


@pytest.fixture
def day0():
    return datetime(2024, 1, 1, 8, 0)


@pytest.fixture
def days(day0):
    return lambda n: day0 + timedelta(days=n)
