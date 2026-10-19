import numpy as np
import pandas as pd
import pytest

from race_predictor.domain import RaceMode
from race_predictor.evaluation import chronological_split, evaluate, grouped_splits
from race_predictor.evaluation.breakdown import breakdown, segment_masks
from race_predictor.evaluation.modes import AutoModeRouter
from race_predictor.evaluation.runner import compare_models, protocol_predictions
from race_predictor.features import SEQ_FEATURES, build_race_dataset
from race_predictor.models import (
    ModeRouter,
    RidgeBaseline,
    RiegelBaseline,
    RiegelCorrected,
    race_modes,
    riegel_time,
)


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
    one_outlier = evaluate([100, 100, 100], [101, 102, 160])
    assert one_outlier["median_ape_pct"] == pytest.approx(2.0)
    assert one_outlier["mape_pct"] == pytest.approx(21.0)
    assert np.isfinite(m["nrmse_pct"])


def test_breakdown_segments(races):
    ds = build_race_dataset(races)
    ds.static.loc[[0, 1], "is_triathlon"] = 1
    masks = segment_masks(ds)
    assert masks["stand-alone"].sum() + masks["triathlon"].sum() == len(ds)
    assert masks["10k (stand-alone)"].sum() == len(ds) - 2

    index = np.arange(len(ds))
    table = breakdown(ds, ds.y * 1.1, index).set_index("segment")
    assert table.loc["all", "n"] == len(ds) and table.loc["triathlon", "n"] == 2
    assert table["MAPE_%"].to_numpy() == pytest.approx(10.0)
    assert "triathlon" not in breakdown(ds, ds.y[2:], index[2:])["segment"].to_list()


def test_protocol_predictions_cover_both_protocols(races):
    ds = build_race_dataset(races)
    out = {p: (pred, idx) for p, pred, idx in protocol_predictions(ds, RidgeBaseline, 3)}
    assert set(out) == {"chronological", "grouped-CV"}
    pred, idx = out["grouped-CV"]
    assert sorted(idx) == list(range(len(ds))) and len(pred) == len(ds)


def test_contributions_sum_to_prediction(races):
    ds = build_race_dataset(races)
    for model in (RiegelBaseline().fit(ds), RidgeBaseline().fit(ds)):
        assert model.contributions(ds).sum(axis=1).to_numpy() == pytest.approx(model.predict(ds))


def test_mode_router_uses_the_model_for_the_mode(races):
    ds = build_race_dataset(races)
    router = ModeRouter().fit(ds)
    pred = router.predict(ds)
    assert pred == pytest.approx(RiegelBaseline().fit(ds).predict(ds))
    assert router.name == "Router(road=Riegel)" and router.model_for(RaceMode.ROAD).name == "Riegel"
    assert router.contributions(ds).sum(axis=1).to_numpy() == pytest.approx(pred)
    assert set(race_modes(ds)) == {RaceMode.ROAD}
    ridge = ModeRouter({RaceMode.ROAD: RidgeBaseline}).fit(ds).predict(ds)
    assert ridge == pytest.approx(RidgeBaseline().fit(ds).predict(ds))


def test_mode_router_needs_every_mode():
    with pytest.raises(ValueError):
        ModeRouter({})


def test_riegel_corrected_learns_systematic_slowdown(races):
    ds = build_race_dataset(races)
    ds.y = RiegelBaseline().fit(ds).predict(ds) * 1.08  # every race 8 % slower than Riegel
    pred = RiegelCorrected(alpha=1e-3).fit(ds).predict(ds)
    assert pred == pytest.approx(ds.y, rel=1e-3)


def test_auto_router_picks_from_inner_cv(races):
    ds = build_race_dataset(races)
    auto = AutoModeRouter({"Riegel": RiegelBaseline, "Ridge": RidgeBaseline}, n_splits=3).fit(ds)
    assert set(auto.chosen_) == set(RaceMode)
    assert auto.chosen_[RaceMode.ROAD] == min(
        auto.scores_[RaceMode.ROAD], key=auto.scores_[RaceMode.ROAD].get
    )
    assert np.isfinite(auto.predict(ds)).all()
