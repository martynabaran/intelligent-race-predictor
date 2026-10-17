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
from keras import layers, ops
from sklearn.preprocessing import StandardScaler

from race_predictor.features import SEQ_FEATURES, STATIC_NUMERIC, RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.sequence import MAX_LEN, PAD_VALUE, SequenceModel


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


class NonNegativeMaskedPool(layers.Layer):
    """Masked mean- and max-pooling over time for non-negative (post-ReLU) features.

    ``MaskedPool`` pushes padded steps to -1e9 before the max, which is fine in float but makes the
    int8 interpreter abort. For ``h >= 0``, ``max(h * mask)`` is the same maximum.
    """

    def call(self, inputs):
        h, mask = inputs
        m = ops.expand_dims(mask, -1)
        hm = h * m
        mean = ops.sum(hm, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)
        return ops.concatenate([mean, ops.max(hm, axis=1)], axis=-1)


class CompactSequence(SequenceModel):
    """Tiny int8-ready sequence student (~1.2 k parameters).

    Same preprocessing as :class:`SequenceModel`, but the padding mask is a separate input and
    padded steps are zeros. With ``PAD_VALUE`` (999) inside ``seq_input``, one int8 range would
    have to cover both 999 and the standardised values (~[-3, 3]), collapsing the real data.
    """

    def __init__(
        self,
        width=8,
        dense_units=16,
        learning_rate=2e-3,
        epochs=300,
        patience=30,
        batch_size=16,
        seed=0,
    ):
        super().__init__(
            "tcn",
            width=width,
            dense_units=dense_units,
            dropout=0.0,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            seed=seed,
        )
        self.name = "CompactSequence"

    def inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        seq, static = super().inputs(ds)
        mask = (seq[:, :, 0] != PAD_VALUE).astype("float32")
        return [np.where(mask[:, :, None] > 0, seq, 0.0).astype("float32"), mask, static]

    def _build(self, seq_shape, static_shape):
        seq_in = layers.Input(shape=seq_shape, name="seq_input")
        mask_in = layers.Input(shape=seq_shape[:1], name="mask_input")
        h = layers.Dense(self.width, activation="relu")(seq_in)
        h = layers.Conv1D(self.width, 5, padding="causal", dilation_rate=4, activation="relu")(h)
        h = NonNegativeMaskedPool()([h, mask_in])
        static_in = layers.Input(shape=static_shape, name="static_input")
        s = layers.Dense(self.dense_units, activation="relu")(static_in)
        m = layers.Dense(self.dense_units, activation="relu")(layers.concatenate([h, s]))
        model = keras.Model([seq_in, mask_in, static_in], layers.Dense(1)(m))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mae")
        return model

    def fit(self, ds: RaceDataset) -> CompactSequence:
        keras.utils.set_random_seed(self.seed)
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(
            np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        )
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = min(max(len(s) for s in ds.seqs), MAX_LEN)
        x = self.inputs(ds)
        self.model_ = self._build((self.max_len_, len(SEQ_FEATURES)), (x[2].shape[1],))
        self.model_.fit(
            x,
            self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel(),
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
