import math

import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.training_load import (
    ATL_DAYS,
    CTL_DAYS,
    TYPICAL_HRR,
    activity_trimp,
    load_state,
    load_states,
    state_at,
    trimp_per_minute,
)


def activities(rows, **defaults):
    """Canonical-style activity table; one dict per activity, ``date`` as 'YYYY-MM-DD HH:MM'."""
    base = {
        "athlete": "a",
        "type": "Run",
        "moving_s": 3600.0,
        "elapsed_s": 3600.0,
        "avg_hr": 150.0,
        "max_hr": 190.0,
        **defaults,
    }
    df = pd.DataFrame([{**base, **row} for row in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ----- TRIMP ----------------------------------------------------------------------------------


def test_trimp_per_minute_matches_banister_formula():
    hrr = 0.7
    assert trimp_per_minute(hrr) == pytest.approx(hrr * 0.64 * math.exp(1.92 * hrr))


def test_trimp_of_a_measured_hour_run_by_hand():
    # HRr = (150 - 60) / (190 - 60); 60 minutes at that intensity.
    df = activities([{"date": "2024-01-01 08:00"}])
    hrr = (150 - 60) / (190 - 60)
    result = activity_trimp(df)
    assert result["trimp"].iloc[0] == pytest.approx(60 * hrr * 0.64 * math.exp(1.92 * hrr))
    assert not result["trimp_estimated"].iloc[0]


def test_harder_minutes_weigh_more_than_easy_ones():
    easy = activities([{"date": "2024-01-01", "avg_hr": 120.0}])
    hard = activities([{"date": "2024-01-01", "avg_hr": 175.0}])
    assert activity_trimp(hard)["trimp"].iloc[0] > 2 * activity_trimp(easy)["trimp"].iloc[0]


def test_hr_max_column_is_preferred_over_the_athletes_max_hr():
    df = activities([{"date": "2024-01-01", "hr_max": 200.0}])
    expected = activity_trimp(df.drop(columns="hr_max").assign(max_hr=200.0))
    assert activity_trimp(df)["trimp"].iloc[0] == pytest.approx(expected["trimp"].iloc[0])


def test_hr_max_defaults_to_the_athletes_highest_max_hr():
    df = activities(
        [{"date": "2024-01-01", "max_hr": 170.0}, {"date": "2024-01-02", "max_hr": 190.0}]
    )
    trimp = activity_trimp(df)["trimp"]
    assert trimp.iloc[0] == pytest.approx(trimp.iloc[1])  # both scaled against 190


def test_missing_heart_rate_falls_back_to_typical_intensity_and_is_flagged():
    df = activities([{"date": "2024-01-01", "avg_hr": float("nan")}])
    result = activity_trimp(df)
    assert result["trimp_estimated"].iloc[0]
    assert result["trimp"].iloc[0] == pytest.approx(60 * trimp_per_minute(TYPICAL_HRR["Run"]))


def test_fallback_depends_on_activity_type():
    df = activities(
        [{"date": "2024-01-01", "type": t} for t in ("Run", "Weight Training")], avg_hr=float("nan")
    )
    run, weights = activity_trimp(df)["trimp"]
    assert run > weights


def test_unknown_activity_type_uses_a_default_intensity():
    df = activities([{"date": "2024-01-01", "type": "Kayak", "avg_hr": float("nan")}])
    assert activity_trimp(df)["trimp"].iloc[0] > 0


def test_hr_max_not_above_resting_hr_counts_as_unmeasured():
    df = activities([{"date": "2024-01-01", "max_hr": 55.0}])
    assert activity_trimp(df)["trimp_estimated"].iloc[0]


def test_heart_rate_above_hr_max_is_clipped_to_full_reserve():
    df = activities([{"date": "2024-01-01", "avg_hr": 210.0, "hr_max": 190.0}])
    assert activity_trimp(df)["trimp"].iloc[0] == pytest.approx(60 * trimp_per_minute(1.0))


def test_elapsed_time_is_used_when_moving_time_is_missing():
    df = activities([{"date": "2024-01-01", "moving_s": float("nan"), "elapsed_s": 1800.0}])
    half = activity_trimp(activities([{"date": "2024-01-01", "moving_s": 1800.0}]))
    assert activity_trimp(df)["trimp"].iloc[0] == pytest.approx(half["trimp"].iloc[0])


def test_unknown_duration_contributes_no_load():
    df = activities([{"date": "2024-01-01", "moving_s": float("nan"), "elapsed_s": float("nan")}])
    assert activity_trimp(df)["trimp"].iloc[0] == 0.0


def test_trimp_keeps_the_input_index():
    df = activities([{"date": "2024-01-01"}, {"date": "2024-01-02"}]).set_index(pd.Index([7, 9]))
    assert list(activity_trimp(df).index) == [7, 9]


# ----- daily state ----------------------------------------------------------------------------


def test_state_has_the_documented_contract():
    state = load_state(activities([{"date": "2024-01-01"}]))
    assert list(state.columns) == ["trimp", "ctl", "atl", "tsb"]
    assert state.index.name == "date"


def test_rest_days_are_filled_with_zero_load():
    state = load_state(activities([{"date": "2024-01-01"}, {"date": "2024-01-05"}]))
    assert len(state) == 5
    assert (state["trimp"].iloc[1:4] == 0).all()


def test_several_activities_on_one_day_add_up():
    df = activities([{"date": "2024-01-01 07:00"}, {"date": "2024-01-01 18:00"}])
    single = activity_trimp(df)["trimp"].iloc[0]
    assert load_state(df)["trimp"].iloc[0] == pytest.approx(2 * single)


def test_single_impulse_decays_exponentially_and_fatigue_leaves_first():
    df = activities([{"date": "2024-01-01"}])
    impulse = activity_trimp(df)["trimp"].iloc[0]
    state = load_state(df, end=pd.Timestamp("2024-03-01"))

    assert state["ctl"].iloc[0] == pytest.approx(impulse * (1 - math.exp(-1 / CTL_DAYS)))
    assert state["ctl"].iloc[10] == pytest.approx(state["ctl"].iloc[0] * math.exp(-10 / CTL_DAYS))
    assert state["atl"].iloc[10] == pytest.approx(state["atl"].iloc[0] * math.exp(-10 / ATL_DAYS))
    assert state["tsb"].iloc[0] < 0  # fatigue dominates right after the workout
    assert state["tsb"].iloc[30] > 0  # fatigue is gone, fitness remains


def test_constant_daily_load_converges_to_that_load():
    days = pd.date_range("2024-01-01", periods=300, freq="D")
    df = activities([{"date": d} for d in days])
    daily = activity_trimp(df)["trimp"].iloc[0]
    last = load_state(df).iloc[-1]
    assert last["ctl"] == pytest.approx(daily, rel=1e-3)
    assert last["atl"] == pytest.approx(daily, rel=1e-3)
    assert last["tsb"] == pytest.approx(0, abs=0.05 * daily)


def test_tsb_is_ctl_minus_atl():
    df = activities([{"date": f"2024-01-{d:02d}"} for d in (1, 2, 3, 8, 9, 15)])
    state = load_state(df)
    assert (state["tsb"] - (state["ctl"] - state["atl"])).abs().max() < 1e-12


def test_end_extends_past_the_last_activity_as_rest():
    df = activities([{"date": "2024-01-01"}])
    state = load_state(df, end=pd.Timestamp("2024-01-10 15:00"))
    assert state.index[-1] == pd.Timestamp("2024-01-10")
    assert (state["trimp"].iloc[1:] == 0).all()


def test_end_ignores_later_activities():
    df = activities([{"date": "2024-01-01"}, {"date": "2024-01-20"}])
    state = load_state(df, end=pd.Timestamp("2024-01-10"))
    assert state.index[-1] == pd.Timestamp("2024-01-10")
    assert state["trimp"].iloc[1:].eq(0).all()


def test_end_before_the_first_activity_is_rejected():
    with pytest.raises(DataValidationError, match="before the first activity"):
        load_state(activities([{"date": "2024-02-01"}]), end=pd.Timestamp("2024-01-01"))


def test_empty_table_is_rejected():
    with pytest.raises(DataValidationError, match="empty"):
        load_state(activities([{"date": "2024-01-01"}]).iloc[0:0])


def test_load_state_rejects_several_athletes():
    df = activities([{"date": "2024-01-01"}, {"date": "2024-01-01", "athlete": "b"}])
    with pytest.raises(DataValidationError, match="one athlete"):
        load_state(df)


def test_load_states_yields_each_athlete_separately():
    df = activities(
        [
            {"date": "2024-01-01"},
            {"date": "2024-01-03", "athlete": "b"},
            {"date": "2024-01-04", "athlete": "b"},
        ]
    )
    states = dict(load_states(df))
    assert set(states) == {"a", "b"}
    assert len(states["a"]) == 1 and len(states["b"]) == 2


def test_other_sports_count_towards_load():
    run = activities([{"date": "2024-01-01"}])
    run_and_ride = activities([{"date": "2024-01-01"}, {"date": "2024-01-01", "type": "Ride"}])
    assert load_state(run_and_ride)["trimp"].iloc[0] > load_state(run)["trimp"].iloc[0]


# ----- state_at (no leakage) ------------------------------------------------------------------


def test_state_at_excludes_the_race_day_itself():
    df = activities([{"date": "2024-01-01"}, {"date": "2024-01-10"}])  # big workout on race day
    state = load_state(df)
    before = state_at(state, pd.Timestamp("2024-01-10 09:00"))
    assert before.name == pd.Timestamp("2024-01-09")
    assert before["trimp"] == 0  # the race-day session is not in its own features


def test_state_at_can_include_the_day():
    state = load_state(activities([{"date": "2024-01-01"}, {"date": "2024-01-02"}]))
    assert state_at(state, pd.Timestamp("2024-01-02"), inclusive=True).name == pd.Timestamp(
        "2024-01-02"
    )


def test_state_at_before_any_history_is_rejected():
    state = load_state(activities([{"date": "2024-01-05"}]))
    with pytest.raises(DataValidationError, match="no training-load state"):
        state_at(state, pd.Timestamp("2024-01-05"))
