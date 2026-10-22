"""Is a goal realistic? Compare it with the predicted time and its plausible range."""

from __future__ import annotations

from dataclasses import dataclass

from race_predictor.domain import Goal, Verdict
from race_predictor.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class GoalAssessment:
    """Verdict on a goal and the numbers behind it."""

    verdict: Verdict
    goal_time_s: float
    predicted_time_s: float
    low_s: float  # fast end of the plausible range
    high_s: float  # slow end of the plausible range
    gap_pct: float  # goal vs prediction: negative = the goal is faster than predicted


def assess_goal(
    goal: Goal,
    predicted_time_s: float,
    low_s: float,
    high_s: float,
    weeks_to_race: float = 0.0,
    weekly_gain_pct: float = 0.0,
) -> GoalAssessment:
    """Classify ``goal`` against a prediction with a plausible range.

    * ``realistic``: the goal is as slow as, or slower than, the predicted time;
    * ``ambitious``: faster than predicted but not faster than the fast end of the range;
    * ``unlikely``: faster than the fast end of the range.

    The prediction describes fitness *today*. Training until the race may improve it; pass the
    expected improvement as ``weekly_gain_pct`` (default 0: no assumed improvement, the
    conservative choice, since nothing in the data yet supports a rate) and the prediction and
    range are shortened by ``weekly_gain_pct * weeks_to_race`` percent, capped at 15 %.

    Args:
        goal: The target time.
        predicted_time_s: Predicted time now for the goal's distance.
        low_s: Fast end of the plausible range (``<= predicted_time_s``).
        high_s: Slow end of the plausible range (``>= predicted_time_s``).
        weeks_to_race: Weeks until the race.
        weekly_gain_pct: Assumed improvement per week, in percent of the time.

    Raises:
        DataValidationError: If the range does not bracket the prediction or inputs are negative.
    """
    if not 0 < low_s <= predicted_time_s <= high_s:
        raise DataValidationError("expected 0 < low <= predicted <= high")
    if weeks_to_race < 0 or weekly_gain_pct < 0:
        raise DataValidationError("weeks_to_race and weekly_gain_pct must not be negative")

    factor = 1 - min(weekly_gain_pct * weeks_to_race, MAX_TOTAL_GAIN_PCT) / 100
    predicted, low, high = (t * factor for t in (predicted_time_s, low_s, high_s))
    if goal.time_s >= predicted:
        verdict = Verdict.REALISTIC
    elif goal.time_s >= low:
        verdict = Verdict.AMBITIOUS
    else:
        verdict = Verdict.UNLIKELY
    return GoalAssessment(
        verdict, goal.time_s, predicted, low, high, (goal.time_s / predicted - 1) * 100
    )


MAX_TOTAL_GAIN_PCT = 15.0
