"""Hybrid GRU model from the thesis (sequence branch + static branch), ported to a RaceModel.

This stays the faithful port of the thesis architecture. :mod:`race_predictor.models.sequence`
has the same hybrid layout with a selectable encoder (GRU, TCN or Transformer).

TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from race_predictor.features import SEQ_FEATURES, RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.preprocessing import (
    PAD_VALUE,
    numeric_static,
    pad_scaled_sequences,
    scaled_static,
)

if TYPE_CHECKING:
    import keras


def build_gru_race_model(
    seq_shape: tuple[int, int],
    static_shape: tuple[int],
    hidden_size: int = 16,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 8e-4,
) -> keras.Model:
    """Build and compile the thesis architecture (Bi-GRU x2 + dense static branch).

    Follows ``personalized_model.ipynb``; ``seq_shape`` is ``(max_len, n_features)`` and
    ``static_shape`` is ``(n_static_features,)``.
    """
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        GRU,
        BatchNormalization,
        Bidirectional,
        Dense,
        Dropout,
        Input,
        Masking,
        concatenate,
    )
    from tensorflow.keras.optimizers import Adam

    seq_input = Input(shape=seq_shape, name="seq_input")
    seq_branch = Masking(mask_value=PAD_VALUE)(seq_input)
    seq_branch = Bidirectional(GRU(hidden_size, return_sequences=True, kernel_regularizer="l2"))(
        seq_branch
    )
    seq_branch = Dropout(dropout_rate)(seq_branch)
    seq_branch = BatchNormalization()(seq_branch)
    seq_branch = Bidirectional(GRU(hidden_size, kernel_regularizer="l2"))(seq_branch)
    seq_branch = Dense(dense_units, activation="relu", kernel_regularizer="l2")(seq_branch)
    seq_branch = BatchNormalization()(seq_branch)
    seq_branch = Dropout(dropout_rate)(seq_branch)

    static_input = Input(shape=static_shape, name="static_input")
    static_branch = Dense(dense_units, activation="relu", kernel_regularizer="l2")(static_input)
    static_branch = BatchNormalization()(static_branch)
    static_branch = Dropout(dropout_rate)(static_branch)

    merged = concatenate([seq_branch, static_branch])
    merged = Dense(dense_units, activation="relu", kernel_regularizer="l2")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_rate)(merged)
    out = Dense(1, activation="linear", name="pace_sec_per_km")(merged)

    model = Model([seq_input, static_input], out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae", metrics=["mae"])
    return model


class GRUModel(RaceModel):
    """Thesis hybrid GRU, trained with early stopping on a 20 % validation split.

    ``model_kwargs`` are passed to :func:`build_gru_race_model` (``hidden_size``, ``dense_units``,
    ``dropout_rate``, ``learning_rate``). Fitted attributes: ``model_``, the scalers
    (``seq_scaler_``, ``static_scaler_``, ``y_scaler_``) and ``max_len_``.
    """

    name = "GRU"

    def __init__(
        self,
        epochs: int = 300,
        patience: int = 30,
        batch_size: int = 16,
        seed: int = 0,
        **model_kwargs: Any,
    ) -> None:
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.seed = seed
        self.model_kwargs = model_kwargs

    def _inputs(self, ds: RaceDataset, max_len: int | None = None) -> list[np.ndarray]:
        seq = pad_scaled_sequences(ds.seqs, self.seq_scaler_, max_len or self.max_len_)
        return [seq, scaled_static(ds.static, self.static_scaler_)]

    @override
    def fit(self, ds: RaceDataset) -> GRUModel:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        tf.keras.utils.set_random_seed(self.seed)
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(numeric_static(ds.static))
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = max(len(seq) for seq in ds.seqs)

        inputs = self._inputs(ds)
        targets = self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel()
        self.model_ = build_gru_race_model(
            (self.max_len_, len(SEQ_FEATURES)), (inputs[1].shape[1],), **self.model_kwargs
        )
        self.model_.fit(
            inputs,
            targets,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[EarlyStopping(patience=self.patience, restore_best_weights=True)],
        )
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        scaled = self.model_.predict(self._inputs(ds), verbose=0)
        return np.asarray(self.y_scaler_.inverse_transform(scaled)).ravel()
