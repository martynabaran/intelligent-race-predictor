"""Race-pace models: the :class:`RaceModel` interface, baselines, hybrid and router."""

from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import (
    HistGradientBoostingBaseline,
    LightGBMBaseline,
    RidgeBaseline,
    RiegelBaseline,
    riegel_time,
)
from race_predictor.models.hybrid import RiegelCorrected
from race_predictor.models.router import ModeRouter, race_modes

__all__ = [
    "HistGradientBoostingBaseline",
    "LightGBMBaseline",
    "ModeRouter",
    "RaceModel",
    "RidgeBaseline",
    "RiegelBaseline",
    "RiegelCorrected",
    "race_modes",
    "riegel_time",
]
