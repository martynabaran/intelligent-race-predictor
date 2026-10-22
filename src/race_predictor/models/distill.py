"""Knowledge distillation and a tiny tabular student for embedded deployment.

Distillation here is *target blending*: the student is trained on
``alpha * y_true + (1 - alpha) * teacher_prediction`` (``alpha = 1`` = plain training, no
distillation). For an MAE/MSE loss this is the standard soft-target objective, and it works with
any student that is a :class:`RaceModel`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import keras
import numpy as np
import pandas as pd
from keras import layers, ops
from typing_extensions import override

from race_predictor.features import SEQ_FEATURES, RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.preprocessing import PAD_VALUE
from race_predictor.models.sequence import SequenceModel


class Distilled(RaceModel):
    """Fit ``teacher_factory()``, then train ``student_factory()`` on blended targets.

    ``alpha`` weights the true targets (``1`` = plain training). ``cache`` (a dict shared between
    instances) avoids refitting the same teacher when several students are compared on the same
    training data.
    """

    def __init__(
        self,
        teacher_factory: Callable[[], RaceModel],
        student_factory: Callable[[], RaceModel],
        alpha: float = 0.5,
        cache: dict[int, RaceModel] | None = None,
    ) -> None:
        self.teacher_factory, self.student_factory = teacher_factory, student_factory
        self.alpha = alpha
        self.cache = cache if cache is not None else {}
        self.name = f"distilled(a={alpha})"

    @staticmethod
    def _key(ds: RaceDataset) -> int:
        return int(pd.util.hash_pandas_object(ds.meta, index=False).sum()) ^ len(ds)

    def teacher(self, ds: RaceDataset) -> RaceModel:
        """Return the teacher fitted on ``ds`` (from the cache when already fitted)."""
        key = self._key(ds)
        if key not in self.cache:
            self.cache[key] = self.teacher_factory().fit(ds)
        return self.cache[key]

    @override
    def fit(self, ds: RaceDataset) -> Distilled:
        soft = self.teacher(ds).predict(ds)
        blended = ds.with_targets(self.alpha * ds.y + (1 - self.alpha) * soft)
        self.student_ = self.student_factory().fit(blended)
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.student_.predict(ds)


class TabularMLP(RaceModel):
    """Small MLP on the tabular features.

    Inputs are median-imputed and standardised *outside* the graph (two constants per feature on
    a device). Keeping standardisation inside the graph made full-int8 quantisation lose accuracy,
    because one int8 range cannot represent raw features of very different magnitudes.

    ``units`` are the hidden layer sizes; ``patience`` is the early-stopping patience (20 %
    validation split).
    """

    name = "MLP"

    def __init__(
        self,
        units: Sequence[int] = (16, 8),
        dropout: float = 0.1,
        epochs: int = 300,
        patience: int = 30,
        batch_size: int = 16,
        lr: float = 3e-3,
        seed: int = 0,
    ) -> None:
        self.units, self.dropout = tuple(units), dropout
        self.epochs, self.patience, self.batch_size, self.lr, self.seed = (
            epochs,
            patience,
            batch_size,
            lr,
            seed,
        )

    def inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        """Return the imputed, standardised tabular features as the model's only input."""
        features = ds.static[ds.tabular_columns].to_numpy(dtype=float, copy=True)
        missing = np.isnan(features)
        features[missing] = np.take(self.medians_, np.nonzero(missing)[1])
        return [((features - self.mean_) / self.scale_).astype("float32")]

    @override
    def fit(self, ds: RaceDataset) -> TabularMLP:
        keras.utils.set_random_seed(self.seed)
        raw = ds.static[ds.tabular_columns].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(raw, axis=0)
        self.medians_ = np.where(np.isnan(medians), 0.0, medians)
        self.mean_, self.scale_ = np.zeros_like(medians), np.ones_like(medians)
        filled = self.inputs(ds)[0].astype(float)
        self.mean_ = filled.mean(axis=0)
        self.scale_ = np.where(filled.std(axis=0) > 0, filled.std(axis=0), 1.0)
        features = self.inputs(ds)[0]
        self.y_mean_, self.y_std_ = float(ds.y.mean()), float(ds.y.std() or 1.0)

        input_layer = layers.Input(shape=(features.shape[1],), name="features")
        hidden = input_layer
        for n_units in self.units:
            hidden = layers.Dropout(self.dropout)(
                layers.Dense(n_units, activation="relu", kernel_regularizer="l2")(hidden)
            )
        self.model_ = keras.Model(input_layer, layers.Dense(1)(hidden))
        self.model_.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mae")
        self.model_.fit(
            features,
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
        """Convert standardised model outputs back to pace [s/km]."""
        return np.asarray(scaled).ravel() * self.y_std_ + self.y_mean_

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.to_pace(self.model_.predict(self.inputs(ds)[0], verbose=0))


class NonNegativeMaskedPool(layers.Layer):
    """Masked mean- and max-pooling over time for non-negative (post-ReLU) features.

    ``MaskedPool`` pushes padded steps to -1e9 before the max, which is fine in float but makes the
    int8 interpreter abort. For ``h >= 0``, ``max(h * mask)`` is the same maximum.
    """

    @override
    def call(self, inputs: tuple[Any, Any]) -> Any:
        hidden, mask = inputs
        mask_3d = ops.expand_dims(mask, -1)
        masked = hidden * mask_3d
        mean = ops.sum(masked, axis=1) / ops.maximum(ops.sum(mask_3d, axis=1), 1.0)
        return ops.concatenate([mean, ops.max(masked, axis=1)], axis=-1)


class CompactSequence(SequenceModel):
    """Tiny int8-ready sequence student (~1.2 k parameters).

    Same preprocessing as :class:`SequenceModel`, but the padding mask is a separate input and
    padded steps are zeros. With ``PAD_VALUE`` (999) inside ``seq_input``, one int8 range would
    have to cover both 999 and the standardised values (~[-3, 3]), collapsing the real data.
    """

    def __init__(
        self,
        width: int = 8,
        dense_units: int = 16,
        learning_rate: float = 2e-3,
        epochs: int = 300,
        patience: int = 30,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
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

    @override
    def inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        """Return ``[sequence (padding zeroed), padding mask, static features]``."""
        seq, static = super().inputs(ds)
        mask = (seq[:, :, 0] != PAD_VALUE).astype("float32")
        return [np.where(mask[:, :, None] > 0, seq, 0.0).astype("float32"), mask, static]

    def _build(self, seq_shape: tuple[int, int], static_shape: tuple[int]) -> keras.Model:
        seq_in = layers.Input(shape=seq_shape, name="seq_input")
        mask_in = layers.Input(shape=seq_shape[:1], name="mask_input")
        hidden = layers.Dense(self.width, activation="relu")(seq_in)
        hidden = layers.Conv1D(self.width, 5, padding="causal", dilation_rate=4, activation="relu")(
            hidden
        )
        hidden = NonNegativeMaskedPool()([hidden, mask_in])
        static_in = layers.Input(shape=static_shape, name="static_input")
        static_branch = layers.Dense(self.dense_units, activation="relu")(static_in)
        merged = layers.Dense(self.dense_units, activation="relu")(
            layers.concatenate([hidden, static_branch])
        )
        model = keras.Model([seq_in, mask_in, static_in], layers.Dense(1)(merged))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss="mae")
        return model

    @override
    def fit(self, ds: RaceDataset) -> CompactSequence:
        self._fit_scalers(ds)
        inputs = self.inputs(ds)
        self.model_ = self._build((self.max_len_, len(SEQ_FEATURES)), (inputs[2].shape[1],))
        self._train(inputs, ds)
        return self
