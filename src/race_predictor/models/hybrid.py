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
from typing_extensions import override

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import RidgeBaseline, RiegelBaseline


def log_distance_ratio(ds: RaceDataset) -> np.ndarray:
    """Return ``log(race / reference distance)``; non-finite without a reference run."""
    static = ds.static
    return np.log(static["race_distance_m"].to_numpy() / static["riegel_ref_distance_m"].to_numpy())


class RiegelCorrected(RaceModel):
    """Riegel's prediction times the exponential of a ridge-learned log correction.

    ``alpha`` is the ridge penalty. ``use_reference_type`` adds whether Riegel's reference run was
    a race (``riegel_ref_is_race``) as a feature: Riegel errs far more when it projects a
    training run.
    """

    def __init__(self, alpha: float = 1.0, *, use_reference_type: bool = False) -> None:
        self.alpha = alpha
        self.use_reference_type = use_reference_type
        self.name = "Riegel+correction+ref" if use_reference_type else "Riegel+correction"

    def _x(self, ds: RaceDataset) -> np.ndarray:
        static = ds.static
        distance_km = static["race_distance_m"].to_numpy() / 1000
        columns = [
            log_distance_ratio(ds),
            static["is_triathlon"].to_numpy(float),
            static["race_elevation_gain_m"].to_numpy(float) / distance_km,
        ]
        if self.use_reference_type:
            columns.append(static["riegel_ref_is_race"].to_numpy(float))
        return np.column_stack(columns)

    @override
    def fit(self, ds: RaceDataset) -> RiegelCorrected:
        self.riegel_ = RiegelBaseline().fit(ds)
        self.fallback_ = RidgeBaseline().fit(ds)
        ok = np.isfinite(log_distance_ratio(ds))
        residual = np.log(ds.y[ok] / self.riegel_.predict(ds)[ok])
        self.model_ = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha)
        ).fit(self._x(ds)[ok], residual)
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        ok = np.isfinite(log_distance_ratio(ds))
        pred = self.fallback_.predict(ds)
        if ok.any():
            corrected = self.riegel_.predict(ds)[ok] * np.exp(self.model_.predict(self._x(ds)[ok]))
            pred[ok] = corrected
        return pred

    @override
    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Split the predicted pace [s/km] exactly into its parts.

        The parts are Riegel's (reference-run pace and distance adjustment) plus
        ``model_correction``, or the ridge contributions where there is no reference run. Rows sum
        to the prediction.
        """
        ok = np.isfinite(log_distance_ratio(ds))
        riegel = self.riegel_.contributions(ds)
        riegel["model_correction"] = self.predict(ds) - self.riegel_.predict(ds)
        parts: list[pd.DataFrame] = [riegel[ok]]
        if (~ok).any():
            idx = np.flatnonzero(~ok)
            parts.append(self.fallback_.contributions(ds.subset(idx)).set_index(idx))
        return pd.concat(parts).sort_index().fillna(0.0).reset_index(drop=True)
