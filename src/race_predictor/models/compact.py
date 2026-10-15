"""Compact student model and knowledge distillation from a large sequence model.

The student keeps the hybrid layout of :mod:`race_predictor.models.sequence` (sequence branch +
static branch + regression head) but with a single small causal convolution, so it has roughly
two orders of magnitude fewer parameters and converts cleanly to TFLite / int8.

Unlike the teacher, the student takes the padding mask as a separate input and sees padded steps
as zeros. The teacher marks padding with ``PAD_VALUE`` (999) inside the data, which would dominate
the calibrated input range under int8 quantisation and destroy the precision of real values.

Distillation (Hinton et al., 2015, adapted to regression): the student is trained on
``alpha * y + (1 - alpha) * teacher(x)`` for real races, plus a *transfer set* that only the
teacher labels - each training race re-asked for another standard distance, with small input
noise. With ~170 labelled races, the extra teacher-labelled examples are the main benefit.
"""

from __future__ import annotations

from collections.abc import Callable

import keras
import numpy as np
from keras import layers, ops
from sklearn.preprocessing import StandardScaler

from race_predictor.features import SEQ_FEATURES, STATIC_NUMERIC, RaceDataset
from race_predictor.models.sequence import MAX_LEN, PAD_VALUE, SequenceModel

STANDARD_DISTANCES_M = (5000.0, 10000.0, 21097.5, 42195.0)


class NonNegativeMaskedPool(layers.Layer):
    """Masked mean- and max-pooling over time for non-negative (post-ReLU) features.

    The teacher's ``MaskedPool`` pushes padded steps to -1e9 before the max, which is fine in
    float but breaks int8 calibration. For ``h >= 0``, ``max(h * mask)`` is the same maximum.
    """

    def call(self, inputs):
        h, mask = inputs
        m = ops.expand_dims(mask, -1)
        hm = h * m
        mean = ops.sum(hm, axis=1) / ops.maximum(ops.sum(m, axis=1), 1.0)
        return ops.concatenate([mean, ops.max(hm, axis=1)], axis=-1)


def build_student(seq_shape, static_shape, width=8, dense_units=16, kernel_size=5, lr=2e-3):
    seq_in = layers.Input(shape=seq_shape, name="seq_input")
    mask_in = layers.Input(shape=seq_shape[:1], name="mask_input")
    h = layers.Dense(width, activation="relu")(seq_in)
    h = layers.Conv1D(width, kernel_size, padding="causal", dilation_rate=4, activation="relu")(h)
    h = NonNegativeMaskedPool()([h, mask_in])

    static_in = layers.Input(shape=static_shape, name="static_input")
    s = layers.Dense(dense_units, activation="relu")(static_in)
    m = layers.Dense(dense_units, activation="relu")(layers.concatenate([h, s]))
    model = keras.Model(
        [seq_in, mask_in, static_in], layers.Dense(1, name="pace_sec_per_km")(m)
    )
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return model


def to_student_inputs(teacher_inputs: list[np.ndarray]) -> list[np.ndarray]:
    """[seq with PAD_VALUE, static] -> [seq with zeros, mask, static]."""
    seq, static = teacher_inputs
    mask = (seq[:, :, 0] != PAD_VALUE).astype("float32")
    return [np.where(mask[:, :, None] > 0, seq, 0.0).astype("float32"), mask, static]


class CompactModel(SequenceModel):
    """Small student network trained directly on race labels (the no-distillation control)."""

    def __init__(
        self, width=8, dense_units=16, learning_rate=2e-3, epochs=300, patience=30,
        batch_size=32, seed=0,
    ):
        super().__init__(
            "tcn", width=width, dense_units=dense_units, dropout=0.0,
            learning_rate=learning_rate, epochs=epochs, patience=patience,
            batch_size=batch_size, seed=seed,
        )
        self.name = "Compact"

    def student_inputs(self, ds: RaceDataset) -> list[np.ndarray]:
        return to_student_inputs(SequenceModel.inputs(self, ds))

    def _fit_preprocessing(self, ds: RaceDataset) -> None:
        self.seq_scaler_ = StandardScaler().fit(np.nan_to_num(np.vstack(ds.seqs)))
        self.static_scaler_ = StandardScaler().fit(
            np.nan_to_num(ds.static[STATIC_NUMERIC].to_numpy(float))
        )
        self.y_scaler_ = StandardScaler().fit(ds.y.reshape(-1, 1))
        self.max_len_ = min(max(len(s) for s in ds.seqs), MAX_LEN)

    def _train(self, x: list[np.ndarray], y_scaled: np.ndarray, validation=None) -> None:
        self.model_ = build_student(
            (self.max_len_, len(SEQ_FEATURES)), (x[2].shape[1],), self.width,
            self.dense_units, lr=self.learning_rate,
        )
        fit_kw = dict(validation_data=validation) if validation else dict(validation_split=0.2)
        self.model_.fit(
            x, y_scaled, epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
            verbose=0, **fit_kw,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
            ],
        )

    def fit(self, ds: RaceDataset) -> CompactModel:
        keras.utils.set_random_seed(self.seed)
        self._fit_preprocessing(ds)
        self._train(self.student_inputs(ds), self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel())
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        pred = self.model_.predict(self.student_inputs(ds), verbose=0)
        return self.y_scaler_.inverse_transform(pred).ravel()


def with_race_distance(ds: RaceDataset, distances_m: np.ndarray) -> RaceDataset:
    """Same training history, different race distance (the counterfactual the teacher labels)."""
    static = ds.static.copy()
    static["race_distance_m"] = distances_m
    meta = ds.meta.copy()
    meta["race_distance_m"] = distances_m
    return RaceDataset(ds.seqs, static, ds.y, meta)


class DistilledModel(CompactModel):
    """Compact student distilled from a (larger) sequence model."""

    def __init__(
        self,
        teacher: Callable[[], SequenceModel] = lambda: SequenceModel("tcn"),
        alpha: float = 0.5,
        n_augment: int = 3,
        noise: float = 0.05,
        **student_kw,
    ):
        super().__init__(**student_kw)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.teacher, self.alpha, self.n_augment, self.noise = teacher, alpha, n_augment, noise
        self.name = "Distilled"

    def _transfer_set(self, ds: RaceDataset, rng: np.random.Generator) -> list[np.ndarray]:
        """Teacher inputs for each race re-asked at a random other distance, with noise."""
        parts = []
        for _ in range(self.n_augment):
            dist = rng.choice(STANDARD_DISTANCES_M, size=len(ds))
            seq, static = self.teacher_.inputs(with_race_distance(ds, dist))
            real = seq[:, :, :1] != PAD_VALUE
            seq = np.where(real, seq + rng.normal(0, self.noise, seq.shape), seq)
            static = static + rng.normal(0, self.noise, static.shape)
            parts.append([seq.astype("float32"), static.astype("float32")])
        return [np.concatenate(p) for p in zip(*parts, strict=True)]

    def fit(self, ds: RaceDataset, teacher: SequenceModel | None = None) -> DistilledModel:
        """Fit the teacher (or reuse an already fitted one) and distil it into the student."""
        self.teacher_ = teacher if teacher is not None else self.teacher().fit(ds)
        keras.utils.set_random_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        # Share the teacher's preprocessing so teacher outputs are on the student's target scale.
        for attr in ("seq_scaler_", "static_scaler_", "y_scaler_", "max_len_"):
            setattr(self, attr, getattr(self.teacher_, attr))

        x_real = self.teacher_.inputs(ds)
        y_real = self.y_scaler_.transform(ds.y.reshape(-1, 1)).ravel()
        soft_real = self.teacher_.model_.predict(x_real, verbose=0).ravel()

        # Hold out real races (not augmented copies) for early stopping.
        val = rng.random(len(ds)) < 0.2
        val[0] = val[0] or not val.any()
        train_x = [a[~val] for a in x_real]
        train_y = self.alpha * y_real[~val] + (1 - self.alpha) * soft_real[~val]
        if self.n_augment:
            x_aug = self._transfer_set(ds.subset(np.flatnonzero(~val)), rng)
            train_x = [np.concatenate([a, b]) for a, b in zip(train_x, x_aug, strict=True)]
            train_y = np.concatenate(
                [train_y, self.teacher_.model_.predict(x_aug, verbose=0).ravel()]
            )
        validation = (to_student_inputs([a[val] for a in x_real]), y_real[val])
        self._train(to_student_inputs(train_x), train_y, validation)
        return self
