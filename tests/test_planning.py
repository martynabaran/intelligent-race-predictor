from datetime import date

import pytest

from race_predictor.domain import Goal, Verdict
from race_predictor.exceptions import DataValidationError
from race_predictor.planning import (
    assess_goal,
    equivalent_time,
    training_zones,
    vdot,
    zones_from_race,
)
from race_predictor.planning.feasibility import MAX_TOTAL_GAIN_PCT
from race_predictor.planning.zones import MARATHON_M, oxygen_cost, pace_at, velocity_at

RACE_DAY = date(2027, 4, 12)


# ----- VDOT and zones -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distance_m", "time_s", "expected"),
    [(5_000, 20 * 60, 49.8), (10_000, 40 * 60, 51.9), (42_195, 3 * 3600, 53.5)],
)
def test_vdot_matches_published_equivalents(distance_m, time_s, expected):
    assert vdot(distance_m, time_s) == pytest.approx(expected, abs=0.1)


def test_a_faster_race_means_a_higher_vdot():
    assert vdot(10_000, 2_400) > vdot(10_000, 2_700)


@pytest.mark.parametrize(("distance_m", "time_s"), [(0, 100), (-5, 100), (5_000, 0), (5_000, -1)])
def test_vdot_rejects_non_positive_input(distance_m, time_s):
    with pytest.raises(DataValidationError):
        vdot(distance_m, time_s)


def test_velocity_is_the_inverse_of_oxygen_cost():
    for velocity in (150.0, 250.0, 330.0):
        assert velocity_at(oxygen_cost(velocity)) == pytest.approx(velocity)


def test_equivalent_time_round_trips_through_vdot():
    value = vdot(10_000, 2_700)
    assert equivalent_time(value, 10_000) == pytest.approx(2_700, rel=1e-6)


def test_equivalent_time_grows_with_distance_and_shrinks_with_fitness():
    value = vdot(10_000, 2_700)
    assert (
        equivalent_time(value, 5_000)
        < equivalent_time(value, 10_000)
        < equivalent_time(value, 21_097.5)
    )
    assert equivalent_time(value + 5, 10_000) < equivalent_time(value, 10_000)


def test_equivalent_marathon_is_close_to_the_published_riegel_projection():
    time_10k = 2_700.0
    riegel = time_10k * (MARATHON_M / 10_000) ** 1.06
    assert equivalent_time(vdot(10_000, time_10k), MARATHON_M) == pytest.approx(riegel, rel=0.03)


def test_zones_are_ordered_from_easy_to_repetition():
    zones = zones_from_race(10_000, 2_700)
    assert list(zones) == ["easy", "marathon", "threshold", "interval", "repetition"]
    pace = [z.slow_s_km for z in zones.values()]
    assert pace == sorted(pace, reverse=True)  # each zone is faster than the one before


def test_each_zone_has_its_slow_end_not_faster_than_its_fast_end():
    for zone in zones_from_race(10_000, 2_700).values():
        assert zone.slow_s_km >= zone.fast_s_km


def test_marathon_zone_is_the_marathon_equivalent_pace():
    value = vdot(10_000, 2_700)
    expected = equivalent_time(value, MARATHON_M) / (MARATHON_M / 1000)
    marathon = training_zones(value)["marathon"]
    assert marathon.slow_s_km == marathon.fast_s_km == pytest.approx(expected)


def test_fitter_runner_has_faster_zones_everywhere():
    slow, fast = zones_from_race(10_000, 3_000), zones_from_race(10_000, 2_400)
    for name in slow:
        assert fast[name].slow_s_km < slow[name].slow_s_km
        assert fast[name].fast_s_km < slow[name].fast_s_km


def test_pace_at_full_vo2max_is_between_threshold_and_repetition():
    value = vdot(10_000, 2_700)
    zones = training_zones(value)
    assert zones["interval"].slow_s_km > pace_at(value, 1.0) > zones["repetition"].slow_s_km


def test_zone_paces_are_physiologically_plausible_for_a_recreational_runner():
    zones = zones_from_race(10_000, 2_700)  # 45:00 10 km
    assert 5 * 60 < zones["easy"].fast_s_km < zones["easy"].slow_s_km < 7 * 60
    assert 4 * 60 < zones["marathon"].slow_s_km < 5.5 * 60
    assert 3 * 60 < zones["repetition"].fast_s_km < zones["repetition"].slow_s_km < 4.2 * 60


# ----- goal -----------------------------------------------------------------------------------


def test_goal_pace():
    assert Goal(10_000, 2_700, RACE_DAY).pace_s_per_km == pytest.approx(270)


@pytest.mark.parametrize(("distance_m", "time_s"), [(0, 100), (-1, 100), (10_000, 0), (10_000, -5)])
def test_goal_rejects_non_positive_values(distance_m, time_s):
    with pytest.raises(DataValidationError):
        Goal(distance_m, time_s, RACE_DAY)


def test_goal_is_immutable():
    goal = Goal(10_000, 2_700, RACE_DAY)
    with pytest.raises(AttributeError):
        goal.time_s = 1.0  # type: ignore[misc]


# ----- feasibility ----------------------------------------------------------------------------

PREDICTED, LOW, HIGH = 2_700.0, 2_400.0, 3_000.0


def goal(seconds: float) -> Goal:
    return Goal(10_000, seconds, RACE_DAY)


@pytest.mark.parametrize(
    ("goal_s", "verdict"),
    [
        (3_300, Verdict.REALISTIC),  # slower than predicted
        (2_700, Verdict.REALISTIC),  # exactly the prediction
        (2_699, Verdict.AMBITIOUS),
        (2_400, Verdict.AMBITIOUS),  # at the fast end of the range
        (2_399, Verdict.UNLIKELY),
        (1_800, Verdict.UNLIKELY),
    ],
)
def test_verdict_boundaries(goal_s, verdict):
    assert assess_goal(goal(goal_s), PREDICTED, LOW, HIGH).verdict is verdict


def test_gap_is_negative_when_the_goal_is_faster_than_predicted():
    assessment = assess_goal(goal(2_430), PREDICTED, LOW, HIGH)
    assert assessment.gap_pct == pytest.approx(-10.0)
    assert assessment.predicted_time_s == PREDICTED and (assessment.low_s, assessment.high_s) == (
        LOW,
        HIGH,
    )


def test_expected_training_gain_can_turn_an_ambitious_goal_realistic():
    ambitious = assess_goal(goal(2_600), PREDICTED, LOW, HIGH)
    with_training = assess_goal(
        goal(2_600), PREDICTED, LOW, HIGH, weeks_to_race=12, weekly_gain_pct=0.5
    )
    assert ambitious.verdict is Verdict.AMBITIOUS
    assert with_training.verdict is Verdict.REALISTIC
    assert with_training.predicted_time_s == pytest.approx(PREDICTED * 0.94)
    assert with_training.low_s == pytest.approx(LOW * 0.94)


def test_gain_is_capped():
    assessment = assess_goal(
        goal(2_000), PREDICTED, LOW, HIGH, weeks_to_race=100, weekly_gain_pct=1.0
    )
    assert assessment.predicted_time_s == pytest.approx(PREDICTED * (1 - MAX_TOTAL_GAIN_PCT / 100))


def test_no_assumed_improvement_by_default():
    assert (
        assess_goal(goal(2_600), PREDICTED, LOW, HIGH, weeks_to_race=26).predicted_time_s
        == PREDICTED
    )


@pytest.mark.parametrize(
    ("low", "predicted", "high"), [(2_800, 2_700, 3_000), (2_400, 2_700, 2_600), (0, 2_700, 3_000)]
)
def test_range_must_bracket_the_prediction(low, predicted, high):
    with pytest.raises(DataValidationError, match="low <= predicted <= high"):
        assess_goal(goal(2_600), predicted, low, high)


@pytest.mark.parametrize("kwargs", [{"weeks_to_race": -1}, {"weekly_gain_pct": -0.1}])
def test_negative_weeks_or_gain_are_rejected(kwargs):
    with pytest.raises(DataValidationError, match="must not be negative"):
        assess_goal(goal(2_600), PREDICTED, LOW, HIGH, **kwargs)
