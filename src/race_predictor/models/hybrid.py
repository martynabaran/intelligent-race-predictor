"""Riegel formula with a learned correction (residual learning).

The model predicts ``log(true pace / Riegel pace)`` from a few features and multiplies Riegel's
prediction by its exponential. With ``log(D2 / D1)`` as a feature this also learns a different
Riegel exponent. Races without a reference run fall back to ridge regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import RidgeBaseline, RiegelBaseline


def log_distance_ratio(ds: RaceDataset) -> np.ndarray:
    s = ds.static
    return np.log(s["race_distance_m"].to_numpy() / s["riegel_ref_distance_m"].to_numpy())


class RiegelCorrected(RaceModel):
    """``use_reference_type`` adds whether Riegel's reference run was a race
    (``riegel_ref_is_race``): Riegel errs far more when it projects a training run."""

    def __init__(self, alpha: float = 1.0, use_reference_type: bool = False):
        self.alpha = alpha
        self.use_reference_type = use_reference_type
        self.name = "Riegel+correction+ref" if use_reference_type else "Riegel+correction"

    def _x(self, ds: RaceDataset) -> np.ndarray:
        s = ds.static
        km = s["race_distance_m"].to_numpy() / 1000
        cols = [
            log_distance_ratio(ds),
            s["is_triathlon"].to_numpy(float),
            s["race_elevation_gain_m"].to_numpy(float) / km,
        ]
        if self.use_reference_type:
            cols.append(s["riegel_ref_is_race"].to_numpy(float))
        return np.column_stack(cols)

    def fit(self, ds: RaceDataset) -> RiegelCorrected:
        self.riegel_ = RiegelBaseline().fit(ds)
        self.fallback_ = RidgeBaseline().fit(ds)
        ok = np.isfinite(log_distance_ratio(ds))
        residual = np.log(ds.y[ok] / self.riegel_.predict(ds)[ok])
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha)
        ).fit(self._x(ds)[ok], residual)
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        ok = np.isfinite(log_distance_ratio(ds))
        pred = self.fallback_.predict(ds)
        if ok.any():
            corrected = self.riegel_.predict(ds)[ok] * np.exp(self.model_.predict(self._x(ds)[ok]))
            pred[ok] = corrected
        return pred

    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Exact split of the predicted pace (s/km): Riegel's parts (reference-run pace + distance
        adjustment) plus ``model_correction``; ridge contributions where there is no reference.
        Rows sum to the prediction."""
        ok = np.isfinite(log_distance_ratio(ds))
        riegel = self.riegel_.contributions(ds)
        riegel["model_correction"] = self.predict(ds) - self.riegel_.predict(ds)
        parts = [riegel[ok]]
        if (~ok).any():
            idx = np.flatnonzero(~ok)
            parts.append(self.fallback_.contributions(ds.subset(idx)).set_index(idx))
        return pd.concat(parts).sort_index().fillna(0.0).reset_index(drop=True)
