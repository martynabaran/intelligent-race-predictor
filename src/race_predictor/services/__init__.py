"""The API used by the user interfaces (Streamlit app, CLI).

``app/`` and ``cli.py`` import only from here (plus ``domain`` and ``exceptions``), so models,
features and data processing can change without touching the interfaces. Enforced by
``tests/test_architecture.py``.
"""

from race_predictor.services.catalog import available_sources, compare_baselines, load_activities
from race_predictor.services.load import LoadSummary, form_label, training_load
from race_predictor.services.prediction import (
    DISTANCES_M,
    PREDICTION_COLUMNS,
    Prediction,
    RacePredictor,
    format_time,
    prepare_activities,
)

__all__ = [
    "DISTANCES_M",
    "PREDICTION_COLUMNS",
    "LoadSummary",
    "Prediction",
    "RacePredictor",
    "available_sources",
    "compare_baselines",
    "form_label",
    "format_time",
    "load_activities",
    "prepare_activities",
    "training_load",
]
