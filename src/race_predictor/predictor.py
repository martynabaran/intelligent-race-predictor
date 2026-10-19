"""Backward-compatible alias: the prediction service lives in :mod:`race_predictor.services`."""

from race_predictor.services.prediction import (
    DISTANCES_M,
    PACE_BOUNDS_S_KM,
    PREDICTION_COLUMNS,
    Prediction,
    RacePredictor,
    format_time,
    prepare_activities,
)

__all__ = [
    "DISTANCES_M",
    "PACE_BOUNDS_S_KM",
    "PREDICTION_COLUMNS",
    "Prediction",
    "RacePredictor",
    "format_time",
    "prepare_activities",
]
