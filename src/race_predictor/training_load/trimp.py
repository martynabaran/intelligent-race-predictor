"""Training impulse (TRIMP) per activity, after Banister (1991).

``TRIMP = duration [min] * HRr * 0.64 * exp(b * HRr)`` with the heart-rate reserve fraction
``HRr = (avg_hr - hr_rest) / (hr_max - hr_rest)`` clipped to [0, 1]. The exponential weights hard
minutes more than easy ones; ``b`` is 1.92 for men and 1.67 for women (the exports contain no sex,
so the default is the common 1.92).

Activities without usable heart rate (no strap, or ``hr_max`` not above ``hr_rest``) still
carry load. They get a fallback of ``duration * per-minute load of a typical intensity for the
activity type`` and are flagged in ``trimp_estimated`` so later analyses can tell measured from
assumed load. Those typical intensities (:data:`TYPICAL_HRR`) are assumptions, not fitted values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from race_predictor.domain import ActivityType

BANISTER_B = 1.92
DEFAULT_HR_REST = 60.0
#: Typical heart-rate-reserve fraction per activity type, used when heart rate is missing.
TYPICAL_HRR: dict[str, float] = {
    ActivityType.RUN: 0.65,
    ActivityType.RIDE: 0.55,
    ActivityType.SWIM: 0.60,
    ActivityType.WEIGHT_TRAINING: 0.40,
    ActivityType.WORKOUT: 0.50,
}
_FALLBACK_HRR = 0.50


def trimp_per_minute(hrr: np.ndarray | pd.Series, b: float = BANISTER_B) -> Any:
    """Banister load of one minute at heart-rate-reserve fraction ``hrr`` (vectorised)."""
    return hrr * 0.64 * np.exp(b * hrr)


def activity_trimp(
    activities: pd.DataFrame, hr_rest: float = DEFAULT_HR_REST, b: float = BANISTER_B
) -> pd.DataFrame:
    """TRIMP of every activity, aligned to ``activities.index``.

    Uses ``hr_max`` if the table has it, otherwise each athlete's highest ``max_hr``. Activities
    with unknown duration contribute 0.

    Returns:
        DataFrame with ``trimp`` (float) and ``trimp_estimated`` (bool: fallback was used).
    """
    duration_min = activities["moving_s"].fillna(activities["elapsed_s"]) / 60
    hr_max = (
        activities["hr_max"]
        if "hr_max" in activities
        else activities.groupby("athlete")["max_hr"].transform("max")
    )
    reserve = hr_max - hr_rest
    measured = activities["avg_hr"].notna() & reserve.gt(0)

    hrr = ((activities["avg_hr"] - hr_rest) / reserve).clip(0.0, 1.0)
    typical = activities["type"].map(TYPICAL_HRR).fillna(_FALLBACK_HRR)
    intensity = hrr.where(measured, typical)

    trimp = (duration_min * trimp_per_minute(intensity, b)).fillna(0.0)
    return pd.DataFrame({"trimp": trimp, "trimp_estimated": ~measured})
