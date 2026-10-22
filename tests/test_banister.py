import math

import numpy as np
import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.models.baselines import RIEGEL_EXPONENT as MODELS_RIEGEL_EXPONENT
from race_predictor.training_load import (
    BanisterParams,
    equivalent_speed,
    fit_banister,
    rolling_forecast,
)
from race_predictor.training_load.banister import (
    LITERATURE_TAU1,
    LITERATURE_TAU2,
    MIN_RACES_FIXED_TAUS,
    MIN_RACES_FULL,
)
from race_predictor.training_load.performance import RIEGEL_EXPONENT

START = pd.Timestamp("2024-01-01")
TRUE = BanisterParams(p0=3.5, k1=0.004, k2=0.012, tau1=45.0, tau2=8.0)


def day(n: int) -> pd.Timestamp:
    return START + pd.Timedelta(days=n)


def synthetic_athlete(
    n_races: int, params: BanisterParams = TRUE, noise: float = 0.0, seed: int = 0
):
    """Irregular but reproducible training and races whose performance follows ``params``."""
    rng = np.random.default_rng(seed)
    n_days = 40 + 14 * n_races + 20
    load = pd.Series(
        rng.gamma(2.0, 30.0, n_days) * (rng.random(n_days) > 0.25),
        index=pd.date_range(START, periods=n_days, freq="D"),
    )
    race_days = pd.DatetimeIndex(
        [day(40 + 14 * i + int(rng.integers(0, 5))) for i in range(n_races)]
    )
    performance = params.predict(load, race_days) + rng.normal(0, noise, n_races)
    return load, performance.rename("performance")


# ----- performance index ----------------------------------------------------------------------


def test_riegel_exponent_matches_the_models_layer():
    assert RIEGEL_EXPONENT == MODELS_RIEGEL_EXPONENT


def test_a_ten_km_race_is_its_own_equivalent():
    assert equivalent_speed(10_000, 2_500) == pytest.approx(4.0)


def test_equivalent_efforts_at_different_distances_get_the_same_index():
    t5 = 1_200.0
    t10 = t5 * 2**RIEGEL_EXPONENT  # the Riegel-equivalent 10 km time
    assert equivalent_speed(5_000, t5) == pytest.approx(equivalent_speed(10_000, t10))


def test_faster_race_has_a_higher_index():
    assert equivalent_speed(10_000, 2_400) > equivalent_speed(10_000, 2_700)


# ----- the model itself -----------------------------------------------------------------------


def test_prediction_of_a_single_impulse_by_hand():
    load = pd.Series([100.0], index=[day(0)])
    params = BanisterParams(p0=3.0, k1=0.01, k2=0.03, tau1=40.0, tau2=5.0)
    expected = 3.0 + 0.01 * 100 * math.exp(-10 / 40) - 0.03 * 100 * math.exp(-10 / 5)
    assert params.predict(load, pd.DatetimeIndex([day(10)])).iloc[0] == pytest.approx(expected)


def test_load_on_the_race_day_itself_is_ignored():
    params = BanisterParams(3.0, 0.01, 0.03, 40.0, 5.0)
    load = pd.Series([100.0], index=[day(10)])
    assert params.predict(load, pd.DatetimeIndex([day(10)])).iloc[0] == pytest.approx(3.0)


def test_days_without_load_data_count_as_rest():
    params = BanisterParams(3.0, 0.01, 0.03, 40.0, 5.0)
    assert params.predict(pd.Series(dtype=float), pd.DatetimeIndex([day(5)])).iloc[0] == 3.0


def test_fitness_outlasts_fatigue_after_a_hard_block():
    load = pd.Series(100.0, index=pd.date_range(START, periods=14, freq="D"))
    params = BanisterParams(3.0, 0.01, 0.03, 42.0, 7.0)
    right_after, two_weeks_later = (day(14), day(28))
    p = params.predict(load, pd.DatetimeIndex([right_after, two_weeks_later]))
    assert p.iloc[1] > p.iloc[0]  # tapering lets fatigue dissipate faster than fitness


def test_prediction_accepts_unsorted_dates():
    load, _ = synthetic_athlete(4)
    dates = pd.DatetimeIndex([day(90), day(60)])
    assert TRUE.predict(load, dates).iloc[0] == pytest.approx(TRUE.predict(load, dates[:1]).iloc[0])


# ----- fitting --------------------------------------------------------------------------------


def test_full_fit_recovers_the_true_parameters_without_noise():
    load, races = synthetic_athlete(30)
    fit = fit_banister(load, races)
    assert fit.method == "full" and fit.n_races == 30
    assert fit.rmse < 1e-3
    assert fit.params.tau1 == pytest.approx(TRUE.tau1, rel=0.1)
    assert fit.params.tau2 == pytest.approx(TRUE.tau2, rel=0.1)
    assert fit.params.k1 == pytest.approx(TRUE.k1, rel=0.1)
    assert fit.params.k2 == pytest.approx(TRUE.k2, rel=0.1)


def test_full_fit_with_noise_explains_most_of_the_signal():
    load, races = synthetic_athlete(30, noise=0.02)
    fit = fit_banister(load, races)
    assert fit.rmse < 0.03
    assert fit.rmse < races.std()


def test_time_constants_stay_inside_their_bounds_and_in_order():
    load, races = synthetic_athlete(12, noise=0.1, seed=3)
    params = fit_banister(load, races).params
    assert 20 <= params.tau1 <= 80 and 3 <= params.tau2 <= 19
    assert params.tau1 > params.tau2
    assert params.k1 >= 0 and params.k2 >= 0


def test_few_races_use_literature_time_constants():
    load, races = synthetic_athlete(MIN_RACES_FULL - 1)
    fit = fit_banister(load, races)
    assert fit.method == "fixed_taus"
    assert (fit.params.tau1, fit.params.tau2) == (LITERATURE_TAU1, LITERATURE_TAU2)


def test_very_few_races_fall_back_to_the_mean():
    load, races = synthetic_athlete(MIN_RACES_FIXED_TAUS - 1)
    fit = fit_banister(load, races)
    assert fit.method == "mean_only"
    assert fit.params.p0 == pytest.approx(races.mean())
    assert (fit.params.k1, fit.params.k2) == (0.0, 0.0)
    assert (fit.params.predict(load, races.index) == races.mean()).all()


def test_two_races_on_the_same_day_are_averaged():
    load, races = synthetic_athlete(2)
    doubled = pd.concat([races, races + 0.2]).sort_index()
    fit = fit_banister(load, doubled)
    assert fit.n_races == 2
    assert fit.params.p0 == pytest.approx((races + 0.1).mean())


def test_fit_needs_at_least_one_race():
    with pytest.raises(DataValidationError, match="at least one race"):
        fit_banister(pd.Series(dtype=float), pd.Series(dtype=float))


def test_fit_does_not_modify_its_inputs():
    load, races = synthetic_athlete(10)
    load_before, races_before = load.copy(), races.copy()
    fit_banister(load, races)
    pd.testing.assert_series_equal(load, load_before)
    pd.testing.assert_series_equal(races, races_before)


# ----- out-of-sample forecast -----------------------------------------------------------------


def test_rolling_forecast_skips_the_first_races_and_reports_method():
    load, races = synthetic_athlete(12)
    forecast = rolling_forecast(load, races)
    assert len(forecast) == 12 - MIN_RACES_FIXED_TAUS
    assert list(forecast.columns) == ["performance", "forecast", "method"]
    assert forecast["method"].iloc[0] == "fixed_taus" and forecast["method"].iloc[-1] == "full"


def test_rolling_forecast_never_sees_the_race_it_predicts():
    load, races = synthetic_athlete(12, noise=0.02)
    changed = races.copy()
    changed.iloc[-1] += 5.0  # wildly different last race
    base, other = rolling_forecast(load, races), rolling_forecast(load, changed)
    # every forecast, including the one *for* the altered race, ignores that race's value
    pd.testing.assert_series_equal(base["forecast"], other["forecast"])
    assert other["performance"].iloc[-1] != base["performance"].iloc[-1]


def test_rolling_forecast_beats_the_running_mean_on_noiseless_data():
    load, races = synthetic_athlete(25)
    forecast = rolling_forecast(load, races)
    running_mean = [races.iloc[:i].mean() for i in range(MIN_RACES_FIXED_TAUS, len(races))]
    model_error = np.abs(forecast["forecast"] - forecast["performance"]).mean()
    baseline_error = np.abs(np.array(running_mean) - forecast["performance"].to_numpy()).mean()
    assert model_error < baseline_error
