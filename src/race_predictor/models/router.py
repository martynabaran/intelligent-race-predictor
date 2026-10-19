"""Choose the model by race mode (Strategy pattern).

``ModeRouter`` predicts each race with the model for its mode; every model is fitted on all
races. With only road races left (see :class:`~race_predictor.domain.RaceMode`) it holds one
model - Riegel with a learned correction, the most accurate on athletes the models had not seen
(``docs/RESULTS_MODES.md``) - and is the place where future modes (e.g. trail) plug in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from race_predictor.domain import RaceMode
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.hybrid import RiegelCorrected

DEFAULT_MODELS: dict[RaceMode, Callable[[], RaceModel]] = {RaceMode.ROAD: RiegelCorrected}


def _label(factory) -> str:
    return getattr(factory, "name", None) or getattr(factory, "__name__", "model")


def race_modes(ds: RaceDataset) -> np.ndarray:
    """Mode of every race in ``ds`` (all road races for now)."""
    return np.full(len(ds), RaceMode.ROAD, dtype=object)


class ModeRouter(RaceModel):
    def __init__(self, models: Mapping[RaceMode, Callable[[], RaceModel]] | None = None):
        self.models = dict(DEFAULT_MODELS if models is None else models)
        if set(self.models) != set(RaceMode):
            raise ValueError(f"need one model per mode: {[m.value for m in RaceMode]}")

    @property
    def name(self) -> str:
        return "Router(" + ", ".join(f"{m.value}={_label(f)}" for m, f in self.models.items()) + ")"

    def fit(self, ds: RaceDataset) -> ModeRouter:
        # Modes sharing a factory share one fitted model.
        fitted: dict[int, RaceModel] = {}
        self.fitted_ = {}
        for mode, make in self.models.items():
            if id(make) not in fitted:
                fitted[id(make)] = make().fit(ds)
            self.fitted_[mode] = fitted[id(make)]
        return self

    def model_for(self, mode: RaceMode) -> RaceModel:
        return self.fitted_[RaceMode(mode)]

    def predict(self, ds: RaceDataset) -> np.ndarray:
        modes = race_modes(ds)
        pred = np.full(len(ds), np.nan)
        for mode in set(modes):
            idx = np.flatnonzero(modes == mode)
            pred[idx] = self.fitted_[mode].predict(ds.subset(idx))
        return pred

    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Per-race contributions from the model that predicted it (missing columns = 0)."""
        modes = race_modes(ds)
        parts = []
        for mode in set(modes):
            idx = np.flatnonzero(modes == mode)
            parts.append(self.fitted_[mode].contributions(ds.subset(idx)).set_index(idx))
        return pd.concat(parts).sort_index().fillna(0.0).reset_index(drop=True)
