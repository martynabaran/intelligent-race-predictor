import numpy as np
import pandas as pd
import pytest
from tests.test_load_features import activities, training_frame

from race_predictor.evaluation.breakdown import segment_masks
from race_predictor.evaluation.training_load import (
    banister_report,
    load_feature_comparison,
    paired_ape_difference,
)
from race_predictor.features import build_race_dataset
from race_predictor.models import RidgeBaseline, RiegelBaseline


def dataset():
    return build_race_dataset(training_frame())


def test_comparison_scores_each_model_with_and_without_load():
    ds = dataset()
    table, predictions = load_feature_comparison(
        ds, {"Riegel": RiegelBaseline, "Ridge": RidgeBaseline}, n_splits=3
    )
    assert set(table["load"]) == {"without load", "with load"}
    assert set(table["protocol"]) == {"chronological", "grouped-CV"}
    assert set(predictions) == {
        (m, p) for m in ("Riegel", "Ridge") for p in ("chronological", "grouped-CV")
    }


def test_models_that_ignore_tabular_features_are_unaffected_by_the_switch():
    ds = dataset()
    _, predictions = load_feature_comparison(ds, {"Riegel": RiegelBaseline}, n_splits=3)
    without, with_load, _ = predictions[("Riegel", "grouped-CV")]
    np.testing.assert_allclose(without, with_load)


def test_ridge_sees_the_load_features_when_enabled():
    ds = dataset()
    _, predictions = load_feature_comparison(ds, {"Ridge": RidgeBaseline}, n_splits=3)
    without, with_load, _ = predictions[("Ridge", "grouped-CV")]
    assert not np.allclose(without, with_load)


def test_paired_difference_is_zero_for_identical_predictions():
    ds = dataset()
    index = np.arange(len(ds))
    result = paired_ape_difference(ds, ds.y * 1.1, ds.y * 1.1, index, n_boot=200)
    assert result["mean_change_pp"] == 0 and result["ci_low_pp"] == result["ci_high_pp"] == 0


def test_paired_difference_is_negative_when_load_features_help():
    ds = dataset()
    index = np.arange(len(ds))
    result = paired_ape_difference(ds, ds.y * 1.2, ds.y * 1.05, index, n_boot=300)
    assert result["mean_change_pp"] == pytest.approx(-15.0, abs=0.01)
    assert result["ci_high_pp"] < 0
    assert result["share_improved"] == 1.0


def test_paired_difference_can_be_restricted_to_a_segment():
    ds = dataset()
    index = np.arange(len(ds))
    everything = np.ones(len(ds), dtype=bool)
    only_first = everything.copy()
    only_first[1:] = False
    result = paired_ape_difference(ds, ds.y, ds.y * 1.1, index, segment=only_first, n_boot=50)
    assert result["n"] == 1
    assert paired_ape_difference(ds, ds.y, ds.y, index, segment=everything, n_boot=50)["n"] == len(
        ds
    )
    assert "stand-alone" in segment_masks(ds)  # sanity: masks used by the experiment exist


def test_paired_difference_interval_widens_with_few_athletes():
    ds = dataset()
    rng = np.random.default_rng(0)
    noisy = ds.y * (1 + rng.normal(0, 0.1, len(ds)))
    index = np.arange(len(ds))
    result = paired_ape_difference(ds, ds.y * 1.1, noisy, index, n_boot=300)
    assert result["ci_low_pp"] < result["mean_change_pp"] < result["ci_high_pp"]


def test_banister_report_on_cleaned_activities():
    df = training_frame()
    table, summary = banister_report(df, min_races=3)
    assert set(table["athlete"]) == {"a", "b", "c"}
    assert {"tau1", "tau2", "method", "forecasts"} <= set(table.columns)
    assert summary["athletes"] == 3
    assert table["forecasts"].eq(0).all()  # forecasting needs earlier races first
    assert np.isnan(summary["banister_MAPE_%"])


def test_banister_report_with_enough_races_forecasts_out_of_sample():
    rows = []
    for i in range(80):
        rows.append(
            {"date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=3 * i), "hr_max": 190.0}
        )
    for j in range(10):
        day = pd.Timestamp("2023-03-01") + pd.Timedelta(days=21 * j)
        rows.append(
            {
                "date": day,
                "distance_m": 10_000.0,
                "moving_s": 2_800.0 - 10 * j,
                "is_race": True,
                "hr_max": 190.0,
            }
        )
    df = activities(rows, is_race=False, is_triathlon=0)
    table, summary = banister_report(df, min_races=5)
    assert len(table) == 1 and table["forecasts"].iloc[0] == 7
    for key in ("banister_MAPE_%", "running_mean_MAPE_%", "previous_race_MAPE_%"):
        assert summary[key] >= 0
    assert summary["forecasts"] == 7


def test_banister_report_skips_athletes_with_too_few_races():
    # every athlete in the fixture has 3 race days, so requiring 4 leaves nobody
    table, summary = banister_report(training_frame(), min_races=4)
    assert table.empty
    assert summary["athletes"] == 0 and np.isnan(summary["banister_MAPE_%"])
