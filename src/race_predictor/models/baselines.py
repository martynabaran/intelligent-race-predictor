"""Baselines: Riegel formula, ridge regression, gradient boosting (LightGBM / scikit-learn)."""

from __future__ import annotations

from typing import Any, Self, TypeVar

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel

RIEGEL_EXPONENT = 1.06

_Number = TypeVar("_Number", float, np.ndarray, pd.Series)


def riegel_time(
    ref_time_s: _Number,
    ref_distance_m: _Number,
    target_distance_m: _Number,
    exponent: float = RIEGEL_EXPONENT,
) -> _Number:
    """Riegel (1981): T2 = T1 * (D2 / D1) ** 1.06 (exponent configurable).

    Works on scalars, arrays or Series; the result has the type of the inputs.
    """
    return ref_time_s * (target_distance_m / ref_distance_m) ** exponent


class RiegelBaseline(RaceModel):
    """Project the athlete's fastest >=5 km training run to the race distance.

    Falls back to the training-set mean pace when no reference run exists.
    """

    name = "Riegel"

    def __init__(self, exponent: float = RIEGEL_EXPONENT) -> None:
        self.exponent = exponent
        self._fallback = np.nan

    @override
    def fit(self, ds: RaceDataset) -> RiegelBaseline:
        self._fallback = float(ds.y.mean())
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        static = ds.static
        distance_m = static["race_distance_m"].to_numpy()
        time_s = riegel_time(
            static["riegel_ref_time_s"].to_numpy(),
            static["riegel_ref_distance_m"].to_numpy(),
            distance_m,
            self.exponent,
        )
        pace = time_s / (distance_m / 1000.0)
        return np.where(np.isnan(pace), self._fallback, pace)

    @override
    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Split the predicted pace [s/km] exactly into reference pace and distance adjustment.

        ``(baseline)`` is the training mean pace when there is no reference run.
        """
        static = ds.static
        ref_pace = static["riegel_ref_time_s"].to_numpy() / (
            static["riegel_ref_distance_m"].to_numpy() / 1000
        )
        pace = self.predict(ds)
        missing = np.isnan(ref_pace)
        return pd.DataFrame(
            {
                "(baseline)": np.where(missing, pace, 0.0),
                "best_run_pace": np.where(missing, 0.0, ref_pace),
                "riegel_distance_adjustment": np.where(missing, 0.0, pace - ref_pace),
            }
        )


class _TabularModel(RaceModel):
    """Base of the scikit-learn style models fitted on the tabular feature matrix."""

    def _estimator(self) -> Any:  # sklearn is untyped
        raise NotImplementedError

    def _x(self, ds: RaceDataset) -> np.ndarray:
        features = ds.static[ds.tabular_columns].to_numpy(dtype=float, copy=True)
        if hasattr(self, "empty_"):
            features[:, self.empty_] = 0.0
        return features

    @override
    def fit(self, ds: RaceDataset) -> Self:
        # Columns that are entirely NaN in the training data break some estimators; zero them.
        self.empty_ = np.isnan(ds.static[ds.tabular_columns].to_numpy(dtype=float)).all(axis=0)
        self.model_ = self._estimator().fit(self._x(ds), ds.y)
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        return np.asarray(self.model_.predict(self._x(ds)))


class RidgeBaseline(_TabularModel):
    """Ridge regression on median-imputed, standardised tabular features (``alpha`` = penalty)."""

    name = "Ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    @override
    def _estimator(self) -> Any:  # sklearn is untyped
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha))

    @override
    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Split each prediction exactly into per-feature contributions.

        Contributions are in s/km of pace (positive = slower) relative to the average training
        race; ``(baseline)`` is the intercept. Rows sum to the prediction.
        """
        imputer, scaler, ridge = (self.model_[i] for i in range(3))
        standardised = scaler.transform(imputer.transform(self._x(ds)))
        out = pd.DataFrame(standardised * ridge.coef_, columns=ds.tabular_columns)
        out["(baseline)"] = float(ridge.intercept_)
        return out


class LightGBMBaseline(_TabularModel):
    """LightGBM gradient boosting with small, fixed hyper-parameters (optional dependency)."""

    name = "LightGBM"

    @override
    def _estimator(self) -> Any:  # sklearn is untyped
        from lightgbm import LGBMRegressor  # optional dependency (needs libomp on macOS)

        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=8,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )


class HistGradientBoostingBaseline(_TabularModel):
    """scikit-learn gradient boosting; stand-in when LightGBM is unavailable."""

    name = "HistGB"

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 200,
        max_leaf_nodes: int = 8,
        min_samples_leaf: int = 5,
        l2_regularization: float = 0.0,
    ) -> None:
        self.params: dict[str, Any] = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
            "l2_regularization": l2_regularization,
        }

    @override
    def _estimator(self) -> Any:  # sklearn is untyped
        return HistGradientBoostingRegressor(random_state=0, **self.params)
