"""Knowledge distillation and a tiny tabular student for embedded deployment.

Distillation here is *target blending*: the student is trained on
``alpha * y_true + (1 - alpha) * teacher_prediction`` (``alpha = 1`` = plain training, no
distillation). For an MAE/MSE loss this is the standard soft-target objective, and it works with
any student that is a :class:`RaceModel`.
"""

from __future__ import annotations

import keras
import numpy as np
import pandas as pd
from keras import layers

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel


class Distilled(RaceModel):
    """Fit ``teacher_factory()``, then train ``student_factory()`` on blended targets.

    ``cache`` (a dict shared between instances) avoids refitting the same teacher when several
    students are compared on the same training data.
    """

    def __init__(
        self, teacher_factory, student_factory, alpha: float = 0.5, cache: dict | None = None
    ):
        self.teacher_factory, self.student_factory = teacher_factory, student_factory
        self.alpha = alpha
        self.cache = cache if cache is not None else {}
        self.name = f"distilled(a={alpha})"

    @staticmethod
    def _key(ds: RaceDataset) -> int:
        return int(pd.util.hash_pandas_object(ds.meta, index=False).sum()) ^ len(ds)

    def teacher(self, ds: RaceDataset) -> RaceModel:
        key = self._key(ds)
        if key not in self.cache:
            self.cache[key] = self.teacher_factory().fit(ds)
        return self.cache[key]

    def fit(self, ds: RaceDataset) -> Distilled:
        soft = self.teacher(ds).predict(ds)
        blended = RaceDataset(
            ds.seqs, ds.static, self.alpha * ds.y + (1 - self.alpha) * soft, ds.meta
        )
        self.student_ = self.student_factory().fit(blended)
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.student_.predict(ds)


class TabularMLP(RaceModel):
    """Small MLP on the tabular features.

    Inputs are median-imputed and standardised *outside* the graph (two constants per feature on
    a device). Keeping standardisation inside the graph made full-int8 quantisation lose accuracy,
    because one int8 range cannot represent raw features of very different magnitudes.
    """

    name = "MLP"

    def __init__(
        self, units=(16, 8), dropout=0.1, epochs=300, patience=30, batch_size=16, lr=3e-3, seed=0
    ):
        self.units, self.dropout = tuple(units), dropout
        self.epochs, self.patience, self.batch_size, self.lr, self.seed = (
            epochs,
            patience,
            batch_size,
            lr,
            seed,
        )

    def inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        x = ds.static[ds.tabular_columns].to_numpy(dtype=float, copy=True)
        nan = np.isnan(x)
        x[nan] = np.take(self.medians_, np.nonzero(nan)[1])
        return [((x - self.mean_) / self.scale_).astype("float32")]

    def fit(self, ds: RaceDataset) -> TabularMLP:
        keras.utils.set_random_seed(self.seed)
        raw = ds.static[ds.tabular_columns].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(raw, axis=0)
        self.medians_ = np.where(np.isnan(med), 0.0, med)
        self.mean_, self.scale_ = np.zeros_like(med), np.ones_like(med)
        filled = self.inputs(ds)[0].astype(float)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = np.where(filled.std(axis=0) > 0, filled.std(axis=0), 1.0)
        x = self.inputs(ds)[0]
        self.y_mean_, self.y_std_ = float(ds.y.mean()), float(ds.y.std() or 1.0)

        inp = layers.Input(shape=(x.shape[1],), name="features")
        h = inp
        for u in self.units:
            h = layers.Dropout(self.dropout)(
                layers.Dense(u, activation="relu", kernel_regularizer="l2")(h)
            )
        self.model_ = keras.Model(inp, layers.Dense(1)(h))
        self.model_.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mae")
        self.model_.fit(
            x,
            (ds.y - self.y_mean_) / self.y_std_,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
            ],
        )
        return self

    def to_pace(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled).ravel() * self.y_std_ + self.y_mean_

    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.to_pace(self.model_.predict(self.inputs(ds)[0], verbose=0))
