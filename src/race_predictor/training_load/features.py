"""Training-load features for one race: the athlete's state just before it."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from race_predictor.exceptions import DataValidationError
from race_predictor.training_load.state import state_at

LOAD_FEATURES = ["load_ctl", "load_atl", "load_tsb", "load_ctl_ramp_4w"]
RAMP_DAYS = 28


def load_features(state: pd.DataFrame, race_date: pd.Timestamp) -> dict[str, float]:
    """Fitness, fatigue, form and recent fitness change on the eve of ``race_date``.

    Uses only days *before* the race (see :func:`state_at`). ``load_ctl_ramp_4w`` is CTL minus
    CTL four weeks earlier; before the first recorded day CTL counts as 0, consistent with the
    state starting from zero. All NaN if ``state`` has no day before the race.

    Args:
        state: Daily load state of the athlete (:func:`load_state`).
        race_date: Date of the race.
    """
    try:
        eve = state_at(state, race_date)
    except DataValidationError:
        return dict.fromkeys(LOAD_FEATURES, np.nan)
    earlier = cast("pd.Timestamp", eve.name) - pd.Timedelta(days=RAMP_DAYS)
    ctl_before = state["ctl"].get(earlier, 0.0) if earlier >= state.index[0] else 0.0
    return {
        "load_ctl": float(eve["ctl"]),
        "load_atl": float(eve["atl"]),
        "load_tsb": float(eve["tsb"]),
        "load_ctl_ramp_4w": float(eve["ctl"] - ctl_before),
    }
