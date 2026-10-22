"""Daily training-load state: fitness (CTL), fatigue (ATL) and form (TSB).

``CTL`` and ``ATL`` are exponentially weighted averages of the daily TRIMP with time constants of
42 and 7 days (the usual defaults), ``TSB = CTL - ATL`` ("freshness": positive after rest, negative
in a heavy block). Rest days count as zero load, so the series is reindexed to every calendar day.

The averages start from zero, so the first ~:data:`WARMUP_DAYS` days underestimate fitness; use a
state only after that much history. When building features for a race, take the state of the days
*before* it with :func:`state_at` - the race itself must not leak into its own features.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from race_predictor.exceptions import DataValidationError
from race_predictor.training_load.trimp import DEFAULT_HR_REST, activity_trimp

CTL_DAYS = 42
ATL_DAYS = 7
WARMUP_DAYS = CTL_DAYS


def _exponential_average(daily: pd.Series, time_constant_days: float) -> pd.Series:
    """EWMA with decay ``exp(-1/tau)`` per day, starting from an all-zero state.

    ``ewm(adjust=False)`` seeds the average with the first observation, which would treat the
    first workout as already-built fitness; a leading zero day makes the seed really zero.
    """
    padded = pd.Series(np.concatenate([[0.0], daily.to_numpy()]))
    average = padded.ewm(alpha=1 - np.exp(-1 / time_constant_days), adjust=False).mean()
    return pd.Series(average.to_numpy()[1:], index=daily.index)


def load_state(
    activities: pd.DataFrame,
    end: pd.Timestamp | None = None,
    ctl_days: float = CTL_DAYS,
    atl_days: float = ATL_DAYS,
    hr_rest: float = DEFAULT_HR_REST,
) -> pd.DataFrame:
    """Daily load state of one athlete.

    Args:
        activities: Canonical activity table of a single athlete (all activity types count).
        end: Last day to include, e.g. a race date. Later activities are ignored and days after
            the last activity count as rest. Defaults to the day of the last activity.
        ctl_days: Time constant of fitness (CTL).
        atl_days: Time constant of fatigue (ATL).
        hr_rest: Resting heart rate used for TRIMP.

    Returns:
        DataFrame indexed by calendar ``date`` with columns ``trimp, ctl, atl, tsb``.

    Raises:
        DataValidationError: If the table is empty or holds more than one athlete.
    """
    if activities.empty:
        raise DataValidationError("cannot compute training load of an empty activity table")
    if activities["athlete"].nunique() > 1:
        raise DataValidationError("load_state expects one athlete; use load_states for several")

    load = activity_trimp(activities, hr_rest)["trimp"]
    daily = load.groupby(activities["date"].dt.normalize()).sum()
    first, last = daily.index.min(), daily.index.max()
    if end is not None:
        last = pd.Timestamp(end).normalize()
        if last < first:
            raise DataValidationError(f"end {last:%Y-%m-%d} is before the first activity")
    # Reindexing to [first, last] zero-fills rest days, extends past the final activity, and
    # drops activities after ``end``.
    calendar = pd.date_range(first, last, freq="D", name="date")
    daily = daily.reindex(calendar, fill_value=0.0).rename("trimp")

    ctl = _exponential_average(daily, ctl_days)
    atl = _exponential_average(daily, atl_days)
    return pd.DataFrame({"trimp": daily, "ctl": ctl, "atl": atl, "tsb": ctl - atl})


def load_states(activities: pd.DataFrame, **kwargs) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield ``(athlete, load_state)`` for every athlete in the table, one at a time."""
    for athlete, group in activities.groupby("athlete"):
        yield str(athlete), load_state(group, **kwargs)


def state_at(state: pd.DataFrame, date: pd.Timestamp, inclusive: bool = False) -> pd.Series:
    """State on the last day before ``date`` (or on ``date`` itself if ``inclusive``).

    Raises:
        DataValidationError: If ``state`` has no day that early.
    """
    cutoff = pd.Timestamp(date).normalize()
    if not inclusive:
        cutoff -= pd.Timedelta(days=1)
    earlier = state.loc[:cutoff]
    if earlier.empty:
        raise DataValidationError(f"no training-load state before {cutoff:%Y-%m-%d}")
    return earlier.iloc[-1]
