"""Sequence architectures for race prediction: GRU, TCN and a small Transformer encoder.

All three share the same hybrid layout as the thesis model: a sequence encoder over the preceding
365 days of runs, a dense branch for static features, and a regression head. Only the encoder
differs, so differences in results are attributable to the architecture.

Padding is post-padding with ``PAD_VALUE`` in every channel; encoders derive the validity mask
from the first channel.
"""

from __future__ import annotations

import keras
import numpy as np
from keras import layers, ops
from sklearn.preprocessing import StandardScaler

from race_predictor.features import SEQ_FEATURES, STATIC_BINARY, STATIC_NUMERIC, RaceDataset
from race_predictor.models.base import RaceModel

PAD_VALUE = 999.0
MAX_LEN = 256


class PadMask(layers.Layer):
    """(B, T, F) -> (B, T) float mask, 1 for real steps."""

    def call(self, x):
        return ops.cast(ops.not_equal(x[:, :, 0], PAD_VALUE), "float32")


class ApplyMask(layers.Layer):
    """Zero the padded steps: (x, mask) -> x * mask."""

    def call(self, inputs):
        x, mask = inputs
        return x * mask[:, :, None]


class MaskedPool(layers.Layer):
    """Concatenate masked mean- and max-pooling over time."""

    def call(self, inputs):
        h, mask = inputs
        m = mask[:, :, None]
        mean = ops.sum(h * m, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)
        mx = ops.max(h + (m - 1.0) * 1e9, axis=1)
        return ops.concatenate([mean, mx], axis=-1)


def gru_encoder(x, mask, width, dropout):
    h = layers.Masking(mask_value=PAD_VALUE)(x)
    h = layers.Bidirectional(layers.GRU(width, return_sequences=True, kernel_regularizer="l2"))(h)
    h = layers.BatchNormalization()(layers.Dropout(dropout)(h))
    return layers.Bidirectional(layers.GRU(width, kernel_regularizer="l2"))(h)


def tcn_encoder(x, mask, width, dropout, kernel_size=3, n_blocks=7):
    """Residual dilated causal convolutions (dilations 1, 2, 4 ... 64), masked pooling."""
    h = layers.Dense(width)(ApplyMask()([x, mask]))
    for i in range(n_blocks):
        r = h
        for _ in range(2):
            h = layers.Conv1D(
                width, kernel_size, padding="causal", dilation_rate=2**i, activation="relu"
            )(h)
            h = layers.Dropout(dropout)(h)
        h = layers.LayerNormalization()(layers.Add()([r, h]))
        h = ApplyMask()([h, mask])
    return MaskedPool()([h, mask])


def transformer_encoder(x, mask, width, dropout, n_layers=2, n_heads=2, max_len=MAX_LEN):
    h = layers.Dense(width)(ApplyMask()([x, mask]))
    positions = layers.Embedding(max_len, width)(ops.arange(max_len))
    h = h + positions[None, : h.shape[1], :]
    attn_mask = layers.Lambda(lambda m: m[:, None, :] * m[:, :, None])(mask)
    for _ in range(n_layers):
        a = layers.MultiHeadAttention(n_heads, width // n_heads, dropout=dropout)(
            h, h, attention_mask=attn_mask
        )
        h = layers.LayerNormalization()(layers.Add()([h, layers.Dropout(dropout)(a)]))
        f = layers.Dense(width * 2, activation="relu")(h)
        f = layers.Dropout(dropout)(layers.Dense(width)(f))
        h = layers.LayerNormalization()(layers.Add()([h, f]))
        h = ApplyMask()([h, mask])
    return MaskedPool()([h, mask])


ENCODERS = {"gru": gru_encoder, "tcn": tcn_encoder, "transformer": transformer_encoder}


def build_model(encoder, seq_shape, static_shape, width=32, dense_units=32, dropout=0.3, lr=8e-4):
    seq_in = layers.Input(shape=seq_shape, name="seq_input")
    mask = PadMask()(seq_in)
    x = ENCODERS[encoder](seq_in, mask, width, dropout)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(x)
    x = layers.Dropout(dropout)(layers.BatchNormalization()(x))

    static_in = layers.Input(shape=static_shape, name="static_input")
    s = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(static_in)
    s = layers.Dropout(dropout)(layers.BatchNormalization()(s))

    m = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(
        layers.concatenate([x, s])
    )
    m = layers.Dropout(dropout)(layers.BatchNormalization()(m))
    model = keras.Model([seq_in, static_in], layers.Dense(1, name="pace_sec_per_km")(m))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mae")
    return model


class SequenceModel(RaceModel):
    """Hybrid sequence + static model; encoder is ``gru``, ``tcn`` or ``transformer``."""

    def __init__(
        self,
        encoder="tcn",
        width=32,
        dense_units=32,
        dropout=0.3,
        learning_rate=8e-4,
        epochs=200,
        patience=20,
        batch_size=16,
        seed=0,
    ):
        if encoder not in ENCODERS:
            raise ValueError(f"encoder must be one of {sorted(ENCODERS)}")
        self.encoder, self.width, self.dense_units, self.dropout = (
            encoder,
            width,
            dense_units,
            dropout,
        )
        self.learning_rate, self.epochs, self.patience = learning_rate, epochs, patience
        self.batch_size, self.seed = batch_size, seed
        self.name = {"gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}[encoder]

    def inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        from keras.utils import pad_sequences

        scaled = [self.seq_scaler_.transform(np.nan_to_num(s)) for s in ds.seqs]
        seq = pad_sequences(
            scaled,
            maxlen=self.max_len_,
            padding="post",
            truncating="pre",
            value=PAD_VALUE,
            dtype="float32",
        )
        num = np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        static = np.column_stack(
            [self.static_scaler_.transform(num), ds.static[STATIC_BINARY].to_numpy(float)]
        )
        return [seq, np.nan_to_num(static).astype("float32")]

    def fit(self, ds: RaceDataset) -> SequenceModel:
        keras.utils.set_random_seed(self.seed)
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(
            np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        )
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = min(max(len(s) for s in ds.seqs), MAX_LEN)

        x = self.inputs(ds)
        self.model_ = build_model(
            self.encoder,
            (self.max_len_, len(SEQ_FEATURES)),
            (x[1].shape[1],),
            self.width,
            self.dense_units,
            self.dropout,
            self.learning_rate,
        )
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

    def to_pace(self, scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler_.inverse_transform(np.asarray(scaled).reshape(-1, 1)).ravel()

    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.to_pace(self.model_.predict(self.inputs(ds), verbose=0))
