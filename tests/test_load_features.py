import math

import numpy as np
import pandas as pd
import pytest

from load_helpers import activities, history, training_frame
from race_predictor.exceptions import DataValidationError
from race_predictor.features import build_query_dataset, build_race_dataset
from race_predictor.services import LoadSummary, form_label, training_load
from race_predictor.training_load import (
    LOAD_FEATURES,
    activity_trimp,
    load_features,
    load_state,
)

# ----- load_features --------------------------------------------------------------------------


def test_features_are_the_state_on_the_eve_of_the_race():
    df = history(60)
    state = load_state(df)
    race = pd.Timestamp("2024-02-15 09:00")
    features = load_features(state, race)
    eve = state.loc["2024-02-14"]
    assert features["load_ctl"] == pytest.approx(eve["ctl"])
    assert features["load_atl"] == pytest.approx(eve["atl"])
    assert features["load_tsb"] == pytest.approx(eve["tsb"])
    assert set(features) == set(LOAD_FEATURES)


def test_ramp_is_ctl_now_minus_ctl_four_weeks_earlier():
    state = load_state(history(100))
    features = load_features(state, pd.Timestamp("2024-04-01"))
    eve = pd.Timestamp("2024-03-31")
    assert features["load_ctl_ramp_4w"] == pytest.approx(
        state.loc[eve, "ctl"] - state.loc[eve - pd.Timedelta(days=28), "ctl"]
    )


def test_ramp_of_a_young_history_counts_earlier_fitness_as_zero():
    state = load_state(history(20))
    features = load_features(state, pd.Timestamp("2024-01-15"))
    assert features["load_ctl_ramp_4w"] == pytest.approx(features["load_ctl"])


def test_features_are_nan_without_any_earlier_day():
    state = load_state(history(10))
    features = load_features(state, pd.Timestamp("2024-01-01"))
    assert all(math.isnan(v) for v in features.values())


def test_a_rest_block_raises_form_and_lowers_fitness_slowly():
    train = activities([{"date": d} for d in pd.date_range("2024-01-01", periods=60)])
    state = load_state(train, end=pd.Timestamp("2024-03-31"))  # 1 month of rest after
    hard, rested = (
        load_features(state, pd.Timestamp("2024-03-01")),
        load_features(state, pd.Timestamp("2024-03-31")),
    )
    assert rested["load_tsb"] > hard["load_tsb"]
    assert 0 < rested["load_ctl"] < hard["load_ctl"]


# ----- dataset integration --------------------------------------------------------------------


def test_dataset_carries_load_features_but_hides_them_by_default():
    ds = build_race_dataset(training_frame())
    assert set(LOAD_FEATURES) <= set(ds.static.columns)
    assert not set(LOAD_FEATURES) & set(ds.tabular_columns)
    assert set(LOAD_FEATURES) <= set(ds.with_load().tabular_columns)
    assert ds.static[LOAD_FEATURES].notna().all().all()


def test_the_switch_survives_subsetting_and_new_targets():
    ds = build_race_dataset(training_frame()).with_load()
    assert ds.subset([0, 1]).include_load
    assert ds.with_targets(ds.y * 2).include_load
    assert not ds.with_load(False).include_load


def test_with_targets_replaces_only_the_targets():
    ds = build_race_dataset(training_frame())
    new = ds.with_targets(ds.y + 1)
    assert (new.y == ds.y + 1).all()
    assert new.static is ds.static and new.meta is ds.meta


def test_races_do_not_see_training_after_them():
    df = training_frame()
    base = build_race_dataset(df)
    race = base.meta["race_date"].iloc[0]
    athlete = base.meta["athlete"].iloc[0]
    extra = activities(
        [
            {"athlete": athlete, "date": race + pd.Timedelta(days=d), "moving_s": 20_000.0}
            for d in (1, 5)
        ],
        is_race=False,
        is_triathlon=0,
        hr_max=190.0,
    )
    changed = build_race_dataset(pd.concat([df, extra], ignore_index=True))
    first = lambda ds: ds.static.iloc[0][LOAD_FEATURES].to_numpy(dtype=float)  # noqa: E731
    np.testing.assert_allclose(first(changed), first(base))


def test_other_sports_raise_the_load_before_a_race():
    df = training_frame()
    race = df[df["is_race"] & (df["athlete"] == "a")]["date"].min()
    rides = activities(
        [
            {"athlete": "a", "date": race - pd.Timedelta(days=d), "type": "Ride"}
            for d in range(1, 20)
        ],
        is_race=False,
        is_triathlon=0,
        hr_max=190.0,
    )
    with_rides = build_race_dataset(pd.concat([df, rides], ignore_index=True))
    base = build_race_dataset(df)
    key = lambda ds: ds.static[ds.meta["athlete"].eq("a")].iloc[0]["load_atl"]  # noqa: E731
    assert key(with_rides) > key(base)


def test_query_dataset_has_load_features_that_ignore_the_race_day_and_later():
    df = history(150)
    race_date = pd.Timestamp("2024-05-01")
    ds = build_query_dataset(df, [10_000.0], race_date)
    later = pd.concat([df, activities([{"date": "2024-05-10", "moving_s": 40_000.0}])])
    ds_later = build_query_dataset(later, [10_000.0], race_date)
    assert ds.static[LOAD_FEATURES].notna().all().all()
    pd.testing.assert_frame_equal(ds.static[LOAD_FEATURES], ds_later.static[LOAD_FEATURES])


def test_query_in_the_future_treats_the_gap_as_rest():
    df = history(150)
    soon = build_query_dataset(df, [10_000.0], df["date"].max() + pd.Timedelta(days=1))
    later = build_query_dataset(df, [10_000.0], df["date"].max() + pd.Timedelta(days=21))
    assert later.static["load_atl"].iloc[0] < soon.static["load_atl"].iloc[0]
    assert later.static["load_tsb"].iloc[0] > soon.static["load_tsb"].iloc[0]


def test_load_features_are_consistent_between_training_and_query_paths():
    df = training_frame()
    ds = build_race_dataset(df)
    athlete, race = ds.meta["athlete"].iloc[0], ds.meta["race_date"].iloc[0]
    own = df[df["athlete"] == athlete].drop(columns=["is_race", "is_triathlon", "hr_max"])
    query = build_query_dataset(own, [float(ds.meta["race_distance_m"].iloc[0])], race)
    np.testing.assert_allclose(
        query.static[LOAD_FEATURES].to_numpy(dtype=float),
        ds.static[LOAD_FEATURES].iloc[[0]].to_numpy(dtype=float),
    )


# ----- service --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tsb", "label"),
    [(20.0, "fresh"), (5.1, "fresh"), (5.0, "neutral"), (-10.0, "neutral"), (-30.0, "fatigued")],
)
def test_form_label_thresholds(tsb, label):
    assert form_label(tsb) == label


def test_training_load_returns_state_and_summary():
    state, summary = training_load(history(100))
    assert isinstance(summary, LoadSummary)
    assert summary.ctl == pytest.approx(state["ctl"].iloc[-1])
    assert summary.tsb == pytest.approx(summary.ctl - summary.atl)
    assert summary.form == form_label(summary.tsb)
    assert summary.estimated_share == 0.0


def test_training_load_reports_the_share_that_rests_on_the_fallback():
    df = history(60)
    df.loc[df.index[:30], "avg_hr"] = np.nan  # no heart rate for the first half
    _, summary = training_load(df)
    expected = activity_trimp(df)
    share = expected.loc[expected["trimp_estimated"], "trimp"].sum() / expected["trimp"].sum()
    assert summary.estimated_share == pytest.approx(share)
    assert 0 < summary.estimated_share < 1


def test_training_load_end_extends_the_series_as_rest():
    state, _ = training_load(history(60), end=pd.Timestamp("2024-04-01"))
    assert state.index[-1] == pd.Timestamp("2024-04-01")
    assert state["tsb"].iloc[-1] > 0


def test_training_load_rejects_empty_input_after_cleaning():
    df = history(5, avg_speed_ms=12.0)  # impossible running speeds are removed by cleaning
    with pytest.raises(DataValidationError, match="no usable activities"):
        training_load(df)
