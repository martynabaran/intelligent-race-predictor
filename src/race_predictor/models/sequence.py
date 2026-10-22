"""Sequence architectures for race prediction: GRU, TCN and a small Transformer encoder.

All three share the same hybrid layout as the thesis model: a sequence encoder over the preceding
365 days of runs, a dense branch for static features, and a regression head. Only the encoder
differs, so differences in results are attributable to the architecture.

Padding is post-padding with ``PAD_VALUE`` in every channel; encoders derive the validity mask
from the first channel.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import keras
import numpy as np
from keras import layers, ops
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from race_predictor.features import SEQ_FEATURES, RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.preprocessing import (
    MAX_LEN,
    PAD_VALUE,
    numeric_static,
    pad_scaled_sequences,
    scaled_static,
)

Tensor = Any  # Keras tensors are untyped (keras has no stubs in this project)
Encoder = Callable[..., Tensor]


class PadMask(layers.Layer):
    """(B, T, F) -> (B, T) float mask, 1 for real steps."""

    @override
    def call(self, x: Tensor) -> Tensor:
        return ops.cast(ops.not_equal(x[:, :, 0], PAD_VALUE), "float32")


class ApplyMask(layers.Layer):
    """Zero the padded steps: (x, mask) -> x * mask."""

    @override
    def call(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        x, mask = inputs
        return x * mask[:, :, None]


class MaskedPool(layers.Layer):
    """Concatenate masked mean- and max-pooling over time."""

    @override
    def call(self, inputs: tuple[Tensor, Tensor]) -> Tensor:
        hidden, mask = inputs
        mask_3d = mask[:, :, None]
        mean = ops.sum(hidden * mask_3d, axis=1) / ops.maximum(ops.sum(mask_3d, axis=1), 1.0)
        maximum = ops.max(hidden + (mask_3d - 1.0) * 1e9, axis=1)
        return ops.concatenate([mean, maximum], axis=-1)


def gru_encoder(x: Tensor, mask: Tensor, width: int, dropout: float) -> Tensor:
    """Two stacked bidirectional GRUs (the thesis encoder); ``mask`` is unused (Keras masking)."""
    del mask  # the Masking layer derives its own mask from PAD_VALUE
    h = layers.Masking(mask_value=PAD_VALUE)(x)
    h = layers.Bidirectional(layers.GRU(width, return_sequences=True, kernel_regularizer="l2"))(h)
    h = layers.BatchNormalization()(layers.Dropout(dropout)(h))
    return layers.Bidirectional(layers.GRU(width, kernel_regularizer="l2"))(h)


def tcn_encoder(
    x: Tensor, mask: Tensor, width: int, dropout: float, kernel_size: int = 3, n_blocks: int = 7
) -> Tensor:
    """Residual dilated causal convolutions (dilations 1, 2, 4 ... 64), masked pooling."""
    h = layers.Dense(width)(ApplyMask()([x, mask]))
    for block in range(n_blocks):
        residual = h
        for _ in range(2):
            h = layers.Conv1D(
                width, kernel_size, padding="causal", dilation_rate=2**block, activation="relu"
            )(h)
            h = layers.Dropout(dropout)(h)
        h = layers.LayerNormalization()(layers.Add()([residual, h]))
        h = ApplyMask()([h, mask])
    return MaskedPool()([h, mask])


def transformer_encoder(
    x: Tensor,
    mask: Tensor,
    width: int,
    dropout: float,
    n_layers: int = 2,
    n_heads: int = 2,
    max_len: int = MAX_LEN,
) -> Tensor:
    """Transformer encoder with learned positions and padding-aware attention, masked pooling."""
    h = layers.Dense(width)(ApplyMask()([x, mask]))
    positions = layers.Embedding(max_len, width)(ops.arange(max_len))
    h = h + positions[None, : h.shape[1], :]
    attn_mask = layers.Lambda(lambda m: m[:, None, :] * m[:, :, None])(mask)
    for _ in range(n_layers):
        attended = layers.MultiHeadAttention(n_heads, width // n_heads, dropout=dropout)(
            h, h, attention_mask=attn_mask
        )
        h = layers.LayerNormalization()(layers.Add()([h, layers.Dropout(dropout)(attended)]))
        feed_forward = layers.Dense(width * 2, activation="relu")(h)
        feed_forward = layers.Dropout(dropout)(layers.Dense(width)(feed_forward))
        h = layers.LayerNormalization()(layers.Add()([h, feed_forward]))
        h = ApplyMask()([h, mask])
    return MaskedPool()([h, mask])


ENCODERS: dict[str, Encoder] = {
    "gru": gru_encoder,
    "tcn": tcn_encoder,
    "transformer": transformer_encoder,
}


def build_model(
    encoder: str,
    seq_shape: tuple[int, int],
    static_shape: tuple[int],
    width: int = 32,
    dense_units: int = 32,
    dropout: float = 0.3,
    lr: float = 8e-4,
) -> keras.Model:
    """Build and compile the hybrid model: ``encoder`` over the sequence plus a static branch.

    ``encoder`` is a key of :data:`ENCODERS`; ``seq_shape`` is ``(max_len, n_features)`` and
    ``static_shape`` is ``(n_static_features,)``. The output is the standardised pace.
    """
    seq_in = layers.Input(shape=seq_shape, name="seq_input")
    mask = PadMask()(seq_in)
    seq_branch = ENCODERS[encoder](seq_in, mask, width, dropout)
    seq_branch = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(seq_branch)
    seq_branch = layers.Dropout(dropout)(layers.BatchNormalization()(seq_branch))

    static_in = layers.Input(shape=static_shape, name="static_input")
    static_branch = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(static_in)
    static_branch = layers.Dropout(dropout)(layers.BatchNormalization()(static_branch))

    merged = layers.Dense(dense_units, activation="relu", kernel_regularizer="l2")(
        layers.concatenate([seq_branch, static_branch])
    )
    merged = layers.Dropout(dropout)(layers.BatchNormalization()(merged))
    model = keras.Model([seq_in, static_in], layers.Dense(1, name="pace_sec_per_km")(merged))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mae")
    return model


class SequenceModel(RaceModel):
    """Hybrid sequence + static model; encoder is ``gru``, ``tcn`` or ``transformer``.

    ``width`` is the encoder width; ``patience`` is the early-stopping patience (20 % validation
    split). Fitted attributes: ``model_``, the scalers (``seq_scaler_``, ``static_scaler_``,
    ``y_scaler_``) and ``max_len_``.
    """

    def __init__(
        self,
        encoder: str = "tcn",
        width: int = 32,
        dense_units: int = 32,
        dropout: float = 0.3,
        learning_rate: float = 8e-4,
        epochs: int = 200,
        patience: int = 20,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
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
        """Return the padded standardised sequences and the standardised static features."""
        seq = pad_scaled_sequences(ds.seqs, self.seq_scaler_, self.max_len_)
        return [seq, scaled_static(ds.static, self.static_scaler_)]

    def _fit_scalers(self, ds: RaceDataset) -> None:
        """Seed the RNGs and fit the input/target scalers and the padded length on ``ds``."""
        keras.utils.set_random_seed(self.seed)
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(numeric_static(ds.static))
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = min(max(len(seq) for seq in ds.seqs), MAX_LEN)

    @override
    def fit(self, ds: RaceDataset) -> SequenceModel:
        self._fit_scalers(ds)
        inputs = self.inputs(ds)
        self.model_ = build_model(
            self.encoder,
            (self.max_len_, len(SEQ_FEATURES)),
            (inputs[1].shape[1],),
            self.width,
            self.dense_units,
            self.dropout,
            self.learning_rate,
        )
        self._train(inputs, ds)
        return self

    def _train(self, inputs: list[np.ndarray], ds: RaceDataset) -> None:
        """Train ``model_`` on ``inputs`` and the standardised targets of ``ds``."""
        self.model_.fit(
            inputs,
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

    def to_pace(self, scaled: np.ndarray) -> np.ndarray:
        """Convert standardised model outputs back to pace [s/km]."""
        return np.asarray(
            self.y_scaler_.inverse_transform(np.asarray(scaled).reshape(-1, 1))
        ).ravel()

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.to_pace(self.model_.predict(self.inputs(ds), verbose=0))
