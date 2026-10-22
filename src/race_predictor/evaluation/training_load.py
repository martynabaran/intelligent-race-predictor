"""Do training-load features help, and how well does the Banister model describe form?

Two questions, both answered out-of-sample:

* :func:`load_feature_comparison` / :func:`paired_ape_difference`: the same models with and
  without the load features (CTL, ATL, TSB, CTL ramp), scored per segment under both protocols,
  with a bootstrap over *athletes* for the difference (races of one athlete are not independent).
* :func:`banister_report`: per athlete, forecast each race from earlier races only and compare
  with two naive forecasts (running mean of earlier races, previous race).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from race_predictor.evaluation.breakdown import breakdown
from race_predictor.evaluation.runner import protocol_predictions
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.training_load import (
    equivalent_speed,
    fit_banister,
    load_state,
    rolling_forecast,
)
from race_predictor.training_load.banister import MIN_RACES_FIXED_TAUS, TAU1_BOUNDS, TAU2_BOUNDS

ModelFactory = Callable[[], RaceModel]


def load_feature_comparison(
    ds: RaceDataset, factories: dict[str, ModelFactory], n_splits: int = 5
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Score every model with and without load features.

    Returns:
        A long table (``model, load, protocol, segment, n, MAPE_%, MedAPE_%, MAE_time_min``) and
        the predictions behind it, keyed by ``(model, protocol)``, as
        ``(pred_without, pred_with, test_index)`` over the same test races (for paired
        comparisons).
    """
    rows: list[pd.DataFrame] = []
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    variants = {"without load": ds.with_load(False), "with load": ds.with_load(True)}
    for name, make in factories.items():
        per_variant: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for variant, data in variants.items():
            for protocol, pred, index in protocol_predictions(data, make, n_splits):
                rows.append(
                    breakdown(data, pred, index).assign(model=name, load=variant, protocol=protocol)
                )
                per_variant.setdefault(protocol, {})[variant] = (pred, index)
        for protocol, pair in per_variant.items():
            (p0, i0), (p1, i1) = pair["without load"], pair["with load"]
            assert (i0 == i1).all(), "both variants must be scored on the same races"
            predictions[(name, protocol)] = (p0, p1, i0)
    return pd.concat(rows, ignore_index=True), predictions


def paired_ape_difference(
    ds: RaceDataset,
    pred_without: np.ndarray,
    pred_with: np.ndarray,
    index: np.ndarray,
    segment: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean change in absolute percentage error when load features are added.

    Negative = load features help. The 95 % interval comes from resampling athletes with
    replacement, so it reflects that an athlete's races are correlated.

    Args:
        ds: The dataset the predictions refer to.
        pred_without / pred_with: Predicted paces for the races in ``index``.
        index: Positions of the scored races in ``ds``.
        segment: Optional boolean mask over ``ds`` restricting the comparison (e.g. stand-alone).
    """
    y = ds.y[index]
    ape_without = np.abs(pred_without - y) / y * 100  # the distance cancels in a ratio
    ape_with = np.abs(pred_with - y) / y * 100
    athletes = ds.meta["athlete"].to_numpy()[index]
    keep = np.ones(len(index), dtype=bool) if segment is None else segment[index]
    diff, athletes = (ape_with - ape_without)[keep], athletes[keep]

    groups = [diff[athletes == a] for a in np.unique(athletes)]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = np.concatenate([groups[i] for i in pick])
        means.append(sample.mean())
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(keep.sum()),
        "mean_change_pp": float(diff.mean()),
        "ci_low_pp": float(low),
        "ci_high_pp": float(high),
        "share_improved": float((diff < 0).mean()),
    }


def _mape(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.abs(forecast - actual) / actual) * 100)


def banister_report(
    cleaned: pd.DataFrame, min_races: int = 5
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-athlete Banister fits and out-of-sample forecasts of race performance.

    Performance is the Riegel-equivalent 10 km speed; errors are relative (%), so they read as a
    race-time error. Athletes with fewer than ``min_races`` detected races are skipped.

    Args:
        cleaned: Activities from ``cleaning_pipeline("train")`` (has ``is_race``, ``hr_max``).
        min_races: Minimum number of distinct race days per athlete.

    Returns:
        Per-athlete table and a pooled summary (forecast-weighted MAPE of the Banister model,
        the running mean and the previous race; share of fits with a time constant at a bound).
    """
    rows = []
    for athlete, group in cleaned.groupby("athlete"):
        races = group[group["is_race"] & (group["type"] == "Run")]
        time = races["moving_s"].fillna(races["elapsed_s"])
        speed = pd.Series(
            equivalent_speed(races["distance_m"].to_numpy(), time.to_numpy()),
            index=races["date"].dt.normalize(),
        )
        speed = speed.replace([np.inf, -np.inf], np.nan).dropna()
        speed = speed.groupby(level=0).mean().sort_index()
        if len(speed) < min_races:
            continue

        trimp = load_state(group)["trimp"]
        fit = fit_banister(trimp, speed)
        forecast = rolling_forecast(trimp, speed)
        row = {
            "athlete": athlete,
            "races": fit.n_races,
            "method": fit.method,
            "tau1": fit.params.tau1,
            "tau2": fit.params.tau2,
            "forecasts": len(forecast),
            "at_bound": fit.method == "full"
            and any(
                np.isclose(value, bound, atol=0.5)
                for value, bounds in (
                    (fit.params.tau1, TAU1_BOUNDS),
                    (fit.params.tau2, TAU2_BOUNDS),
                )
                for bound in bounds
            ),
        }
        if len(forecast):
            actual = forecast["performance"].to_numpy()
            mean_forecast = np.array(
                [speed.iloc[:i].mean() for i in range(MIN_RACES_FIXED_TAUS, len(speed))]
            )
            previous = speed.shift(1).iloc[MIN_RACES_FIXED_TAUS:].to_numpy()
            row |= {
                "banister_MAPE_%": _mape(forecast["forecast"].to_numpy(), actual),
                "running_mean_MAPE_%": _mape(mean_forecast, actual),
                "previous_race_MAPE_%": _mape(previous, actual),
            }
        rows.append(row)

    columns = [
        "athlete", "races", "method", "tau1", "tau2", "forecasts", "at_bound",
        "banister_MAPE_%", "running_mean_MAPE_%", "previous_race_MAPE_%",
    ]  # fmt: skip
    table = pd.DataFrame(rows, columns=columns)
    scored = table.dropna(subset=["banister_MAPE_%"])
    weight = scored["forecasts"]

    def pooled(column: str) -> float:
        return float(np.average(scored[column], weights=weight)) if len(scored) else float("nan")

    summary = {
        "athletes": float(len(table)),
        "forecasts": float(weight.sum()),
        "banister_MAPE_%": pooled("banister_MAPE_%"),
        "running_mean_MAPE_%": pooled("running_mean_MAPE_%"),
        "previous_race_MAPE_%": pooled("previous_race_MAPE_%"),
        "full_fits": float((table["method"] == "full").sum()),
        "full_fits_at_bound": float(table["at_bound"].sum()),
    }
    return table, summary
