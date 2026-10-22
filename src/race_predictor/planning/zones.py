"""Training pace zones from a race result, using Daniels & Gilbert's VDOT model.

A race result (distance, time) is turned into a single fitness number, VDOT, with the oxygen-cost
and %VO2max-sustained formulas of Daniels & Gilbert (1979). The training zones are the velocities
at fixed fractions of VO2max for that VDOT. The fractions in :data:`ZONE_INTENSITY` are close
approximations of Daniels' published tables, not the tables themselves; the marathon zone is the
exact marathon-equivalent pace of the same VDOT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from race_predictor.domain import MARATHON_M
from race_predictor.exceptions import DataValidationError

#: Fractions of VO2max: (slow end, fast end) of each zone.
ZONE_INTENSITY = {
    "easy": (0.59, 0.74),
    "threshold": (0.86, 0.88),
    "interval": (0.97, 1.00),
    "repetition": (1.05, 1.10),
}


@dataclass(frozen=True, slots=True)
class PaceRange:
    """A pace zone in seconds per kilometre; ``slow_s_km >= fast_s_km``."""

    slow_s_km: float
    fast_s_km: float


def oxygen_cost(velocity_m_min: float) -> float:
    """VO2 (ml/kg/min) needed to run at ``velocity_m_min`` (Daniels & Gilbert)."""
    return -4.60 + 0.182258 * velocity_m_min + 0.000104 * velocity_m_min**2


def sustained_fraction(duration_min: float) -> float:
    """Fraction of VO2max an athlete can hold for ``duration_min`` minutes."""
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * duration_min)
        + 0.2989558 * math.exp(-0.1932605 * duration_min)
    )


def vdot(distance_m: float, time_s: float) -> float:
    """Fitness index equivalent to running ``distance_m`` in ``time_s``.

    Raises:
        DataValidationError: If distance or time is not positive.
    """
    if distance_m <= 0 or time_s <= 0:
        raise DataValidationError("distance and time must be positive to compute VDOT")
    minutes = time_s / 60
    return oxygen_cost(distance_m / minutes) / sustained_fraction(minutes)


def velocity_at(vo2: float) -> float:
    """Running velocity (m/min) whose oxygen cost is ``vo2`` (inverse of :func:`oxygen_cost`)."""
    quadratic, linear, constant = 0.000104, 0.182258, -4.60 - vo2
    discriminant = linear * linear - 4 * quadratic * constant
    return (-linear + math.sqrt(discriminant)) / (2 * quadratic)


def pace_at(vdot_value: float, fraction: float) -> float:
    """Pace (s/km) at ``fraction`` of VO2max for a runner with ``vdot_value``."""
    return 60_000 / velocity_at(vdot_value * fraction)


def equivalent_time(vdot_value: float, distance_m: float) -> float:
    """Return the time (s) a runner with ``vdot_value`` needs for ``distance_m``.

    VDOT is monotone in time, so this inverts :func:`vdot` by bisection.
    """
    low, high = 60.0, 100 * 3600.0
    for _ in range(80):
        middle = (low + high) / 2
        if vdot(distance_m, middle) > vdot_value:
            low = middle  # too fast: this time would imply a higher VDOT
        else:
            high = middle
    return (low + high) / 2


def training_zones(vdot_value: float) -> dict[str, PaceRange]:
    """Pace zones (easy, marathon, threshold, interval, repetition) for ``vdot_value``."""
    zones = {
        name: PaceRange(pace_at(vdot_value, slow), pace_at(vdot_value, fast))
        for name, (slow, fast) in ZONE_INTENSITY.items()
    }
    marathon = equivalent_time(vdot_value, MARATHON_M) / (MARATHON_M / 1000)
    zones["marathon"] = PaceRange(marathon, marathon)
    order = ["easy", "marathon", "threshold", "interval", "repetition"]
    return {name: zones[name] for name in order}


def zones_from_race(distance_m: float, time_s: float) -> dict[str, PaceRange]:
    """Pace zones implied by a race (or a prediction) of ``distance_m`` in ``time_s``."""
    return training_zones(vdot(distance_m, time_s))
