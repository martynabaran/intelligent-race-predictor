import numpy as np
import pandas as pd
import pytest

from race_predictor.evaluation import chronological_split, evaluate, grouped_splits
from race_predictor.evaluation.runner import compare_models
from race_predictor.features import SEQ_FEATURES, build_race_dataset
from race_predictor.models import RidgeBaseline, RiegelBaseline, riegel_time


@pytest.fixture
def races(frame):
    """3 athletes, each with 30 weekly easy runs, then 4 races (every 28 days)."""
    rows = []
    for a, speed in zip("abc", (3.0, 3.5, 4.0), strict=True):
        start = pd.Timestamp("2023-01-01")
        for i in range(30):
            rows.append(
                dict(
                    athlete=a,
                    date=start + pd.Timedelta(days=7 * i),
                    distance_m=8000.0,
                    avg_speed_ms=speed,
                    is_race=False,
                    is_triathlon=0,
                )
            )
        for j in range(4):
            rows.append(
                dict(
                    athlete=a,
                    date=start + pd.Timedelta(days=215 + 28 * j),
                    distance_m=10000.0,
                    avg_speed_ms=speed * 1.1,
                    moving_s=10000 / (speed * 1.1),
                    is_race=True,
                    is_triathlon=0,
                )
            )
    return frame(rows)


def test_riegel_formula():
    assert riegel_time(1200, 5000, 10000) == pytest.approx(1200 * 2**1.06)


def test_dataset_one_example_per_race(races):
    ds = build_race_dataset(races)
    assert len(ds) == 12 and ds.meta["athlete"].nunique() == 3
    assert all(s.shape[1] == len(SEQ_FEATURES) for s in ds.seqs)
    assert ds.y[0] == pytest.approx(1000 / (3.0 * 1.1))


def test_dataset_window_excludes_race_day_and_later(races):
    ds = build_race_dataset(races)
    for seq in ds.seqs:
        assert (seq[:, 0] > 0).all()  # days_to_race strictly positive


def test_splits(races):
    ds = build_race_dataset(races)
    train, test = chronological_split(ds)
    assert len(test) == 3  # latest race of each athlete
    assert (
        ds.meta.loc[test]
        .groupby("athlete")["race_date"]
        .min()
        .ge(ds.meta.loc[train].groupby("athlete")["race_date"].max())
        .all()
    )
    for tr, te in grouped_splits(ds, 3):
        assert not set(ds.meta.loc[tr, "athlete"]) & set(ds.meta.loc[te, "athlete"])


def test_models_and_comparison(races):
    ds = build_race_dataset(races)
    table = compare_models(ds, {"Riegel": RiegelBaseline, "Ridge": RidgeBaseline}, n_splits=3)
    assert set(table["protocol"]) == {"chronological", "grouped-CV"}
    assert (table["MAPE_%"] < 30).all()


def test_metrics():
    m = evaluate([100, 200], [110, 180])
    assert m["mae"] == 15 and m["mape_pct"] == pytest.approx(10.0)
    assert np.isfinite(m["nrmse_pct"])
