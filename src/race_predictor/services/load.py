"""Training-load service for the user interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from race_predictor.exceptions import DataValidationError
from race_predictor.processing import Pipeline, shared_steps
from race_predictor.training_load import activity_trimp, load_state

#: Rule-of-thumb TSB thresholds for the form label (a convention, not a measured cut-off).
FRESH_TSB = 5.0
FATIGUED_TSB = -10.0


@dataclass(frozen=True, slots=True)
class LoadSummary:
    """Today's training-load numbers and how much of them rests on assumptions."""

    ctl: float  # fitness
    atl: float  # fatigue
    tsb: float  # form = ctl - atl
    form: str  # "fresh", "neutral" or "fatigued"
    estimated_share: float  # share of the total load that came from the no-heart-rate fallback


def form_label(tsb: float) -> str:
    """Rule-of-thumb reading of TSB: ``fresh`` (> 5), ``fatigued`` (< -10), else ``neutral``."""
    if tsb > FRESH_TSB:
        return "fresh"
    return "fatigued" if tsb < FATIGUED_TSB else "neutral"


def training_load(
    activities: pd.DataFrame, end: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, LoadSummary]:
    """Daily fitness / fatigue / form of one athlete, and a summary of the last day.

    Activities are cleaned with the shared steps only (no imputation, so heart-rate gaps stay
    visible and are priced with the flagged fallback).

    Args:
        activities: Canonical activity table of one athlete.
        end: Last day to include (default: the day of the last activity).

    Returns:
        The daily state (columns ``trimp, ctl, atl, tsb``) and its :class:`LoadSummary`.

    Raises:
        DataValidationError: If nothing is left after cleaning, or several athletes are mixed.
    """
    cleaned = Pipeline(shared_steps()).run(
        activities.assign(athlete=activities["athlete"].astype(str))
    )
    if cleaned.empty:
        raise DataValidationError("no usable activities to compute training load from")
    state = load_state(cleaned, end=end)
    trimp = activity_trimp(cleaned)
    total = trimp["trimp"].sum()
    share = float(trimp.loc[trimp["trimp_estimated"], "trimp"].sum() / total) if total else 0.0
    last = state.iloc[-1]
    return state, LoadSummary(
        float(last["ctl"]), float(last["atl"]), float(last["tsb"]), form_label(last["tsb"]), share
    )
