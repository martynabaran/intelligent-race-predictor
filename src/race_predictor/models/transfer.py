"""Transfer learning: pretrain a GRU encoder on Kaggle marathon data, fine-tune on races.

The encoder only sees the channels shared by both data sources (``SHARED_SEQ_FEATURES``).
Fine-tuning adds the static branch (and the race distance) and trains in two phases: encoder
frozen, then everything with a lower learning rate. ``TransferGRU(pretrained=None)`` has the
identical architecture trained from scratch, so the comparison isolates the effect of pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing_extensions import override

from race_predictor.features import SHARED_SEQ_FEATURES, PretrainData, RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.preprocessing import (
    MAX_LEN,
    PAD_VALUE,
    numeric_static,
    pad_scaled_sequences,
    scaled_static,
)

if TYPE_CHECKING:
    import keras


def _build_encoder(
    n_features: int, max_len: int, hidden_size: int, dense_units: int, dropout_rate: float
) -> keras.Model:
    """Build the Bi-GRU x2 sequence encoder shared by pretraining and fine-tuning."""
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        GRU,
        BatchNormalization,
        Bidirectional,
        Dense,
        Dropout,
        Input,
        Masking,
    )

    inputs = Input(shape=(max_len, n_features), name="seq_input")
    hidden = Masking(mask_value=PAD_VALUE)(inputs)
    hidden = Bidirectional(GRU(hidden_size, return_sequences=True, kernel_regularizer="l2"))(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Bidirectional(GRU(hidden_size, kernel_regularizer="l2"))(hidden)
    hidden = Dense(dense_units, activation="relu", kernel_regularizer="l2")(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(dropout_rate)(hidden)
    return Model(inputs, hidden, name="encoder")


@dataclass
class PretrainedEncoder:
    """A pretrained encoder: its weights, the sequence scaler and the architecture it needs."""

    weights: list[np.ndarray]
    seq_scaler: StandardScaler
    max_len: int
    hidden_size: int
    dense_units: int
    dropout_rate: float

    def save(self, path: str | Path) -> None:
        """Pickle the encoder to ``path`` (joblib)."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> PretrainedEncoder:
        """Load an encoder saved with :meth:`save` (joblib/pickle: only load trusted files)."""
        encoder: PretrainedEncoder = joblib.load(path)  # untyped library returns Any
        return encoder


def pretrain_encoder(
    data: PretrainData,
    hidden_size: int = 16,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    epochs: int = 100,
    patience: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> PretrainedEncoder:
    """Train encoder + linear head to predict marathon pace from the preceding-year sequence.

    ``patience`` is the early-stopping patience (10 % validation split).
    """
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    tf.keras.utils.set_random_seed(seed)
    scaler = StandardScaler().fit(np.nan_to_num(np.vstack(data.seqs)))
    max_len = min(max(len(seq) for seq in data.seqs), MAX_LEN)
    y_scaler = StandardScaler().fit(data.y.reshape(-1, 1))

    encoder = _build_encoder(
        len(SHARED_SEQ_FEATURES), max_len, hidden_size, dense_units, dropout_rate
    )
    model = Model(encoder.input, Dense(1)(encoder.output))
    model.compile(optimizer=Adam(learning_rate), loss="mae")
    model.fit(
        pad_scaled_sequences(data.seqs, scaler, max_len),
        y_scaler.transform(data.y.reshape(-1, 1)).ravel(),
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
    )
    return PretrainedEncoder(
        encoder.get_weights(), scaler, max_len, hidden_size, dense_units, dropout_rate
    )


class TransferGRU(RaceModel):
    """Fine-tune (``pretrained`` given) or train from scratch (``None``) on a RaceDataset.

    With a pretrained encoder, the architecture hyper-parameters come from it (``hidden_size``,
    ``dense_units`` and ``dropout_rate`` apply only to training from scratch). Training runs
    ``freeze_epochs`` with the encoder frozen at ``learning_rate``, then up to ``epochs`` with
    everything trainable at ``finetune_lr`` (early stopping with ``patience``).
    """

    def __init__(
        self,
        pretrained: PretrainedEncoder | None = None,
        hidden_size: int = 16,
        dense_units: int = 32,
        dropout_rate: float = 0.3,
        freeze_epochs: int = 20,
        epochs: int = 200,
        patience: int = 30,
        batch_size: int = 16,
        learning_rate: float = 8e-4,
        finetune_lr: float = 2e-4,
        seed: int = 0,
    ) -> None:
        self.pretrained = pretrained
        self.hidden_size, self.dense_units, self.dropout_rate = (
            hidden_size,
            dense_units,
            dropout_rate,
        )
        self.freeze_epochs, self.epochs, self.patience = freeze_epochs, epochs, patience
        self.batch_size, self.learning_rate, self.finetune_lr = (
            batch_size,
            learning_rate,
            finetune_lr,
        )
        self.seed = seed
        self.name = "GRU-finetuned" if pretrained is not None else "GRU-scratch"

    def _inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        seq = pad_scaled_sequences(ds.shared_seqs(), self.seq_scaler_, self.max_len_)
        return [seq, scaled_static(ds.static, self.static_scaler_)]

    @override
    def fit(self, ds: RaceDataset) -> TransferGRU:
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, concatenate
        from tensorflow.keras.optimizers import Adam

        tf.keras.utils.set_random_seed(self.seed)
        seqs = ds.shared_seqs()
        pre = self.pretrained
        if pre is not None:
            self.seq_scaler_, self.max_len_ = pre.seq_scaler, pre.max_len
            hidden, dense, drop = pre.hidden_size, pre.dense_units, pre.dropout_rate
        else:
            self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(seqs)))
            self.max_len_ = min(max(len(seq) for seq in seqs), MAX_LEN)
            hidden, dense, drop = self.hidden_size, self.dense_units, self.dropout_rate
        self.static_scaler_ = StandardScaler().fit(numeric_static(ds.static))
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))

        encoder = _build_encoder(len(SHARED_SEQ_FEATURES), self.max_len_, hidden, dense, drop)
        if pre is not None:
            encoder.set_weights(pre.weights)

        train_inputs = self._inputs(ds)
        static_in = Input(shape=(train_inputs[1].shape[1],), name="static_input")
        static_branch = Dense(dense, activation="relu", kernel_regularizer="l2")(static_in)
        static_branch = Dropout(drop)(BatchNormalization()(static_branch))
        merged = Dense(dense, activation="relu", kernel_regularizer="l2")(
            concatenate([encoder.output, static_branch])
        )
        merged = Dropout(drop)(BatchNormalization()(merged))
        model = Model([encoder.input, static_in], Dense(1)(merged))
        targets = self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel()

        def run(lr: float, epochs: int, patience: int | None) -> None:
            model.compile(optimizer=Adam(lr), loss="mae")
            callbacks = (
                [EarlyStopping(patience=patience, restore_best_weights=True)] if patience else []
            )
            model.fit(
                train_inputs,
                targets,
                validation_split=0.2,
                epochs=epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=0,
                callbacks=callbacks,
            )

        if pre is not None and self.freeze_epochs:
            encoder.trainable = False
            run(self.learning_rate, self.freeze_epochs, None)
            encoder.trainable = True
            run(self.finetune_lr, self.epochs, self.patience)
        else:
            run(self.learning_rate, self.epochs, self.patience)
        self.model_ = model
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        scaled = self.model_.predict(self._inputs(ds), verbose=0)
        return np.asarray(self.y_scaler_.inverse_transform(scaled)).ravel()
