"""Baselines: Riegel formula, ridge regression, gradient boosting (LightGBM / scikit-learn)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel

RIEGEL_EXPONENT = 1.06


def riegel_time(ref_time_s, ref_distance_m, target_distance_m, exponent=RIEGEL_EXPONENT):
    """Riegel (1981): T2 = T1 * (D2 / D1) ** 1.06."""
    return ref_time_s * (target_distance_m / ref_distance_m) ** exponent


class RiegelBaseline(RaceModel):
    """Project the athlete's fastest >=5 km training run to the race distance.

    Falls back to the training-set mean pace when no reference run exists.
    """

    name = "Riegel"

    def __init__(self, exponent: float = RIEGEL_EXPONENT):
        self.exponent = exponent
        self._fallback = np.nan

    def fit(self, ds: RaceDataset) -> RiegelBaseline:
        self._fallback = float(ds.y.mean())
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        s = ds.static
        dist = s["race_distance_m"].to_numpy()
        t = riegel_time(
            s["riegel_ref_time_s"].to_numpy(),
            s["riegel_ref_distance_m"].to_numpy(),
            dist,
            self.exponent,
        )
        pace = t / (dist / 1000.0)
        return np.where(np.isnan(pace), self._fallback, pace)

    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Exact split of the predicted pace (s/km): the reference run's pace plus Riegel's
        distance adjustment (``(baseline)`` = training mean pace when there is no reference)."""
        s = ds.static
        ref_pace = s["riegel_ref_time_s"].to_numpy() / (
            s["riegel_ref_distance_m"].to_numpy() / 1000
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
    def _estimator(self):
        raise NotImplementedError

    def _x(self, ds: RaceDataset) -> np.ndarray:
        x = ds.static[ds.tabular_columns].to_numpy(dtype=float, copy=True)
        if hasattr(self, "empty_"):
            x[:, self.empty_] = 0.0
        return x

    def fit(self, ds: RaceDataset):
        # Columns that are entirely NaN in the training data break some estimators; zero them.
        self.empty_ = np.isnan(ds.static[ds.tabular_columns].to_numpy(dtype=float)).all(axis=0)
        self.model_ = self._estimator().fit(self._x(ds), ds.y)
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.model_.predict(self._x(ds))


class RidgeBaseline(_TabularModel):
    name = "Ridge"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def _estimator(self):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(self.alpha))

    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Exact per-feature contributions (s/km of pace; + = slower), relative to the average
        training race; ``(baseline)`` = intercept. Rows sum to the prediction."""
        imputer, scaler, ridge = (self.model_[i] for i in range(3))
        z = scaler.transform(imputer.transform(self._x(ds)))
        out = pd.DataFrame(z * ridge.coef_, columns=ds.tabular_columns)
        out["(baseline)"] = float(ridge.intercept_)
        return out


class LightGBMBaseline(_TabularModel):
    name = "LightGBM"

    def _estimator(self):
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
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=8,
        min_samples_leaf=5,
        l2_regularization=0.0,
    ):
        self.params = dict(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
        )

    def _estimator(self):
        return HistGradientBoostingRegressor(random_state=0, **self.params)
