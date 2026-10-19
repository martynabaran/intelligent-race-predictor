"""Choose the model by race mode (Strategy pattern).

The error analysis (``docs/RESULTS_BREAKDOWN.md``) shows that no single model is best everywhere:
the Riegel formula wins on stand-alone road races of unseen athletes, while a learned model
handles the slower run leg of a triathlon. ``ModeRouter`` predicts each race with the model for
its mode; every model is fitted on all races (the triathlon flag is one of its features).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from race_predictor.domain import RaceMode
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import RidgeBaseline, RiegelBaseline

DEFAULT_MODELS: dict[RaceMode, Callable[[], RaceModel]] = {
    RaceMode.ROAD: RiegelBaseline,
    RaceMode.TRIATHLON: RidgeBaseline,
}


def _label(factory) -> str:
    return getattr(factory, "name", None) or getattr(factory, "__name__", "model")


def race_modes(ds: RaceDataset) -> np.ndarray:
    """Mode of every race in ``ds``."""
    tri = ds.static["is_triathlon"].to_numpy() == 1
    return np.where(tri, RaceMode.TRIATHLON, RaceMode.ROAD)


class ModeRouter(RaceModel):
    def __init__(self, models: Mapping[RaceMode, Callable[[], RaceModel]] | None = None):
        self.models = dict(models or DEFAULT_MODELS)
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
