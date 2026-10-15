"""Transfer learning: pretrain a GRU encoder on Kaggle marathon data, fine-tune on races.

The encoder only sees the channels shared by both data sources (``SHARED_SEQ_FEATURES``).
Fine-tuning adds the static branch (and the race distance) and trains in two phases: encoder
frozen, then everything with a lower learning rate. ``TransferGRU(pretrained=None)`` has the
identical architecture trained from scratch, so the comparison isolates the effect of pretraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from race_predictor.features import (
    SHARED_SEQ_FEATURES,
    STATIC_BINARY,
    STATIC_NUMERIC,
    PretrainData,
    RaceDataset,
)
from race_predictor.models.base import RaceModel

PAD_VALUE = 999.0
MAX_LEN = 256


def _build_encoder(n_features, max_len, hidden_size, dense_units, dropout_rate):
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

    inp = Input(shape=(max_len, n_features), name="seq_input")
    x = Masking(mask_value=PAD_VALUE)(inp)
    x = Bidirectional(GRU(hidden_size, return_sequences=True, kernel_regularizer="l2"))(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    x = Bidirectional(GRU(hidden_size, kernel_regularizer="l2"))(x)
    x = Dense(dense_units, activation="relu", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return Model(inp, x, name="encoder")


def _pad(seqs, scaler, max_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    scaled = [scaler.transform(np.nan_to_num(s)) for s in seqs]
    return pad_sequences(
        scaled, maxlen=max_len, padding="post", truncating="pre", value=PAD_VALUE, dtype="float32"
    )


@dataclass
class PretrainedEncoder:
    weights: list[np.ndarray]
    seq_scaler: StandardScaler
    max_len: int
    hidden_size: int
    dense_units: int
    dropout_rate: float

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> PretrainedEncoder:
        return joblib.load(path)


def pretrain_encoder(
    data: PretrainData,
    hidden_size=16,
    dense_units=32,
    dropout_rate=0.3,
    epochs=100,
    patience=10,
    batch_size=64,
    learning_rate=1e-3,
    seed=0,
) -> PretrainedEncoder:
    """Train encoder + linear head to predict marathon pace from the preceding-year sequence."""
    import tensorflow as tf
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    tf.keras.utils.set_random_seed(seed)
    scaler = StandardScaler().fit(np.nan_to_num(np.vstack(data.seqs)))
    max_len = min(max(len(s) for s in data.seqs), MAX_LEN)
    y_scaler = StandardScaler().fit(data.y.reshape(-1, 1))

    encoder = _build_encoder(
        len(SHARED_SEQ_FEATURES), max_len, hidden_size, dense_units, dropout_rate
    )
    model = Model(encoder.input, Dense(1)(encoder.output))
    model.compile(optimizer=Adam(learning_rate), loss="mae")
    model.fit(
        _pad(data.seqs, scaler, max_len),
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
    """Fine-tune (``pretrained`` given) or train from scratch (``None``) on a RaceDataset."""

    def __init__(
        self,
        pretrained: PretrainedEncoder | None = None,
        hidden_size=16,
        dense_units=32,
        dropout_rate=0.3,
        freeze_epochs=20,
        epochs=200,
        patience=30,
        batch_size=16,
        learning_rate=8e-4,
        finetune_lr=2e-4,
        seed=0,
    ):
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

    def _inputs(self, ds: RaceDataset):
        seq = _pad(ds.shared_seqs(), self.seq_scaler_, self.max_len_)
        num = np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        static = np.column_stack(
            [self.static_scaler_.transform(num), ds.static[STATIC_BINARY].to_numpy(float)]
        )
        return [seq, np.nan_to_num(static).astype("float32")]

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
            self.max_len_ = min(max(len(s) for s in seqs), MAX_LEN)
            hidden, dense, drop = self.hidden_size, self.dense_units, self.dropout_rate
        self.static_scaler_ = StandardScaler().fit(
            np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        )
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))

        encoder = _build_encoder(len(SHARED_SEQ_FEATURES), self.max_len_, hidden, dense, drop)
        if pre is not None:
            encoder.set_weights(pre.weights)

        x_train = self._inputs(ds)
        static_in = Input(shape=(x_train[1].shape[1],), name="static_input")
        s = Dense(dense, activation="relu", kernel_regularizer="l2")(static_in)
        s = Dropout(drop)(BatchNormalization()(s))
        m = Dense(dense, activation="relu", kernel_regularizer="l2")(
            concatenate([encoder.output, s])
        )
        m = Dropout(drop)(BatchNormalization()(m))
        model = Model([encoder.input, static_in], Dense(1)(m))
        y = self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel()

        def run(lr: float, epochs: int, patience: int | None):
            model.compile(optimizer=Adam(lr), loss="mae")
            cb = [EarlyStopping(patience=patience, restore_best_weights=True)] if patience else []
            model.fit(
                x_train,
                y,
                validation_split=0.2,
                epochs=epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=0,
                callbacks=cb,
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

    def predict(self, ds: RaceDataset) -> np.ndarray:
        pred = self.model_.predict(self._inputs(ds), verbose=0)
        return self.y_scaler_.inverse_transform(pred).ravel()
