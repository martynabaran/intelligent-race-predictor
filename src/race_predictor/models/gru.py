"""Hybrid GRU model from the thesis (sequence branch + static branch), ported to a RaceModel.

TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from race_predictor.features import SEQ_FEATURES, STATIC_BINARY, STATIC_NUMERIC, RaceDataset
from race_predictor.models.base import RaceModel

PAD_VALUE = 999.0


def build_gru_race_model(
    seq_shape, static_shape, hidden_size=16, dense_units=32, dropout_rate=0.3, learning_rate=8e-4
):
    """Architecture as in ``personalized_model.ipynb`` (Bi-GRU x2 + dense static branch)."""
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
    x = Masking(mask_value=PAD_VALUE)(seq_input)
    x = Bidirectional(GRU(hidden_size, return_sequences=True, kernel_regularizer="l2"))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(hidden_size, kernel_regularizer="l2"))(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    static_input = Input(shape=static_shape, name="static_input")
    s = Dense(dense_units, activation="relu", kernel_regularizer="l2")(static_input)
    s = BatchNormalization()(s)
    s = Dropout(dropout_rate)(s)

    merged = concatenate([x, s])
    merged = Dense(dense_units, activation="relu", kernel_regularizer="l2")(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(dropout_rate)(merged)
    out = Dense(1, activation="linear", name="pace_sec_per_km")(merged)

    model = Model([seq_input, static_input], out)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mae", metrics=["mae"])
    return model


class GRUModel(RaceModel):
    name = "GRU"

    def __init__(self, epochs=300, patience=30, batch_size=16, seed=0, **model_kwargs):
        self.epochs, self.patience, self.batch_size = epochs, patience, batch_size
        self.seed = seed
        self.model_kwargs = model_kwargs

    def _inputs(self, ds: RaceDataset, max_len: int | None = None):
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        scaled = [self.seq_scaler_.transform(np.nan_to_num(s)) for s in ds.seqs]
        seq = pad_sequences(
            scaled,
            maxlen=max_len or self.max_len_,
            padding="post",
            truncating="pre",
            value=PAD_VALUE,
            dtype="float32",
        )
        num = np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        static = np.column_stack(
            [
                self.static_scaler_.transform(num),
                ds.static[STATIC_BINARY].to_numpy(float),
            ]
        )
        return [seq, np.nan_to_num(static).astype("float32")]

    def fit(self, ds: RaceDataset) -> GRUModel:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping

        tf.keras.utils.set_random_seed(self.seed)
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(
            np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        )
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = max(len(s) for s in ds.seqs)

        x = self._inputs(ds)
        y = self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel()
        self.model_ = build_gru_race_model(
            (self.max_len_, len(SEQ_FEATURES)), (x[1].shape[1],), **self.model_kwargs
        )
        self.model_.fit(
            x,
            y,
            validation_split=0.2,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0,
            callbacks=[EarlyStopping(patience=self.patience, restore_best_weights=True)],
        )
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        pred = self.model_.predict(self._inputs(ds), verbose=0)
        return self.y_scaler_.inverse_transform(pred).ravel()
