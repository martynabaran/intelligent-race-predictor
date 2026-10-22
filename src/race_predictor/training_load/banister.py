"""Banister impulse-response model of performance (Banister et al., 1975).

Performance on day ``t`` is a baseline plus a slowly decaying fitness effect minus a quickly
decaying fatigue effect of all *earlier* training::

    p(t) = p0 + k1 * sum_{s<t} w(s) * exp(-(t-s)/tau1) - k2 * sum_{s<t} w(s) * exp(-(t-s)/tau2)

``w`` is the daily load (TRIMP), ``tau1`` (fitness, tens of days) is longer than ``tau2``
(fatigue, days). The load of the race day itself is not used.

For fixed time constants the model is linear in ``(p0, k1, k2)``, so fitting is a one-dimensional
search over ``(tau1, tau2)`` with a non-negative least-squares solve inside. An athlete has few
races, and five parameters overfit quickly, so :func:`fit_banister` degrades gracefully:

* ``>= MIN_RACES_FULL`` races: fit ``tau1``, ``tau2`` too (``method="full"``);
* ``>= MIN_RACES_FIXED_TAUS`` races: literature time constants (42 and 7 days), fit
  ``p0, k1, k2`` (``"fixed_taus"``);
* fewer: the mean performance only, no load effect (``"mean_only"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from race_predictor.exceptions import DataValidationError

LITERATURE_TAU1 = 42.0
LITERATURE_TAU2 = 7.0
TAU1_BOUNDS = (20.0, 80.0)
TAU2_BOUNDS = (3.0, 19.0)  # below TAU1_BOUNDS, so fatigue always decays faster than fitness
MIN_RACES_FULL = 8
MIN_RACES_FIXED_TAUS = 3

FitMethod = Literal["full", "fixed_taus", "mean_only"]


@dataclass(frozen=True)
class BanisterParams:
    """Parameters of the impulse-response model."""

    p0: float
    k1: float
    k2: float
    tau1: float = LITERATURE_TAU1
    tau2: float = LITERATURE_TAU2

    def predict(self, daily_load: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
        """Modelled performance on ``dates`` from earlier load.

        Args:
            daily_load: Load per calendar day (``trimp`` of a load state); days outside its index
                count as rest.
            dates: Days to predict, in any order.
        """
        fitness, fatigue = _impulse_responses(daily_load, dates, self.tau1, self.tau2)
        return pd.Series(
            self.p0 + self.k1 * fitness - self.k2 * fatigue, index=dates, name="performance"
        )


@dataclass(frozen=True)
class BanisterFit:
    """Result of :func:`fit_banister`."""

    params: BanisterParams
    method: FitMethod
    n_races: int
    rmse: float  # in-sample error, in the units of the performance index


def _impulse_responses(
    daily_load: pd.Series, dates: pd.DatetimeIndex, tau1: float, tau2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalised decayed load sums ``sum_{s<t} w(s) exp(-(t-s)/tau)`` for both ``tau``."""
    days = pd.DatetimeIndex(dates).normalize()
    if daily_load.empty:
        return np.zeros(len(days)), np.zeros(len(days))
    load = daily_load.copy()
    load.index = pd.DatetimeIndex(load.index).normalize()
    start = min(load.index.min(), days.min())
    end = max(load.index.max(), days.max())
    calendar = pd.date_range(start, end, freq="D")
    w = load.reindex(calendar, fill_value=0.0).to_numpy(dtype=float)

    target = calendar.get_indexer(days)
    lag = target[:, None] - np.arange(len(calendar))[None, :]  # days between load and target
    past = lag > 0  # strictly earlier days only
    sums = [np.where(past, np.exp(-lag / tau), 0.0) @ w for tau in (tau1, tau2)]
    return sums[0], sums[1]


def _solve_linear(
    fitness: np.ndarray, fatigue: np.ndarray, y: np.ndarray
) -> tuple[BanisterParams, float]:
    """Non-negative ``k1, k2`` and free ``p0`` for given responses (``taus`` filled by caller)."""
    f_mean, g_mean, y_mean = fitness.mean(), fatigue.mean(), y.mean()
    design = np.column_stack([fitness - f_mean, -(fatigue - g_mean)])
    (k1, k2), _ = nnls(design, y - y_mean)
    p0 = y_mean - k1 * f_mean + k2 * g_mean
    residual = y - (p0 + k1 * fitness - k2 * fatigue)
    return BanisterParams(float(p0), float(k1), float(k2)), float(np.sum(residual**2))


def _prepare_races(races: pd.Series) -> pd.Series:
    if races.empty:
        raise DataValidationError("need at least one race to fit the Banister model")
    days = pd.DatetimeIndex(races.index).normalize()
    return races.groupby(days).mean().sort_index()  # several races on one day: average them


def fit_banister(daily_load: pd.Series, races: pd.Series) -> BanisterFit:
    """Fit the model to one athlete's races.

    Args:
        daily_load: Daily load (TRIMP) indexed by date; rest days may be missing.
        races: Performance index (higher = better, e.g. :func:`equivalent_speed`) of each race,
            indexed by race date. Races on the same day are averaged.

    Raises:
        DataValidationError: If ``races`` is empty.
    """
    races = _prepare_races(races)
    y = races.to_numpy(dtype=float)
    n = len(y)
    if n < MIN_RACES_FIXED_TAUS:
        return BanisterFit(
            BanisterParams(float(y.mean()), 0.0, 0.0), "mean_only", n, float(y.std())
        )

    def solve(tau1: float, tau2: float) -> tuple[BanisterParams, float]:
        fitness, fatigue = _impulse_responses(daily_load, races.index, tau1, tau2)
        params, sse = _solve_linear(fitness, fatigue, y)
        return BanisterParams(params.p0, params.k1, params.k2, tau1, tau2), sse

    if n < MIN_RACES_FULL:
        params, sse = solve(LITERATURE_TAU1, LITERATURE_TAU2)
        return BanisterFit(params, "fixed_taus", n, float(np.sqrt(sse / n)))

    # Coarse grid first (the objective is not convex), then a bounded local refinement.
    grid = [(t1, t2) for t1 in (25, 35, 45, 60, 75) for t2 in (4, 7, 10, 14, 18)]
    start = min(grid, key=lambda taus: solve(*taus)[1])
    refined = minimize(
        lambda taus: solve(*taus)[1],
        x0=np.array(start, dtype=float),
        bounds=[TAU1_BOUNDS, TAU2_BOUNDS],
        method="Nelder-Mead",
    )
    params, sse = solve(*refined.x)
    return BanisterFit(params, "full", n, float(np.sqrt(sse / n)))


def rolling_forecast(
    daily_load: pd.Series, races: pd.Series, min_races: int = MIN_RACES_FIXED_TAUS
) -> pd.DataFrame:
    """Honest out-of-sample check: forecast each race from a fit on the *earlier* races only.

    Args:
        daily_load: Daily load (TRIMP) indexed by date.
        races: Performance index per race date.
        min_races: Earlier races needed before a race gets a forecast.

    Returns:
        DataFrame indexed by race date with ``performance``, ``forecast`` and ``method``; races
        with fewer than ``min_races`` predecessors are omitted.
    """
    races = _prepare_races(races)
    rows = []
    for i in range(min_races, len(races)):
        fit = fit_banister(daily_load, races.iloc[:i])
        date = races.index[i : i + 1]
        rows.append(
            {
                "date": races.index[i],
                "performance": races.iloc[i],
                "forecast": float(fit.params.predict(daily_load, date).iloc[0]),
                "method": fit.method,
            }
        )
    return pd.DataFrame(rows, columns=["date", "performance", "forecast", "method"]).set_index(
        "date"
    )
