"""A distance-independent performance index for races.

The Banister model needs one number per race that is comparable across distances. Following the
rest of the project, race times are converted with the Riegel formula to an equivalent 10 km
effort, and the index is that effort's average speed (m/s): higher is better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Riegel exponent. Mirrors ``race_predictor.models.baselines.RIEGEL_EXPONENT`` (a higher layer,
#: so it cannot be imported here); a test keeps the two equal.
RIEGEL_EXPONENT = 1.06
REFERENCE_DISTANCE_M = 10_000.0


def equivalent_speed(
    distance_m: float | np.ndarray | pd.Series, time_s: float | np.ndarray | pd.Series
):
    """Speed (m/s) of the 10 km effort equivalent to running ``distance_m`` in ``time_s``."""
    equivalent_time = time_s * (REFERENCE_DISTANCE_M / distance_m) ** RIEGEL_EXPONENT
    return REFERENCE_DISTANCE_M / equivalent_time
