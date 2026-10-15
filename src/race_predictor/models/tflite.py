"""Export race models to TFLite (float32 / dynamic-range / full int8) and run them.

``TFLiteRaceModel`` wraps a converted model together with the source model's preprocessing, so it
implements :class:`RaceModel` and is evaluated exactly like every other model.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from typing import Literal

import numpy as np

from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.compact import CompactModel
from race_predictor.models.sequence import SequenceModel
from race_predictor.utils import timer

Quantization = Literal["float32", "dynamic", "int8"]
QUANTIZATIONS: tuple[Quantization, ...] = ("float32", "dynamic", "int8")


def to_tflite(
    keras_model,
    quantization: Quantization = "float32",
    representative: list[np.ndarray] | None = None,
) -> bytes:
    """Convert a Keras model. ``int8`` needs ``representative`` inputs for calibration; inputs and
    outputs stay float32 so callers do not have to (de)quantise."""
    import tensorflow as tf

    if quantization not in QUANTIZATIONS:
        raise ValueError(f"quantization must be one of {QUANTIZATIONS}")
    with tempfile.TemporaryDirectory() as tmp:
        keras_model.export(tmp, format="tf_saved_model", verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if quantization != "float32":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization == "int8":
            if representative is None:
                raise ValueError("int8 quantization needs representative inputs")

            # Feed by input name: the SavedModel signature orders inputs alphabetically, not in
            # model order, so a positional list would calibrate the wrong tensors.
            names = [t.name for t in keras_model.inputs]

            def samples():
                for i in range(len(representative[0])):
                    yield {
                        n: a[i : i + 1].astype("float32")
                        for n, a in zip(names, representative, strict=True)
                    }

            converter.representative_dataset = samples
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return converter.convert()


class TFLiteRaceModel(RaceModel):
    """A fitted Keras race model (``SequenceModel`` or a compact student) running on TFLite."""

    def __init__(
        self,
        source: Callable[[], SequenceModel] | SequenceModel,
        quantization: Quantization = "float32",
    ):
        self.source, self.quantization = source, quantization
        self.name = f"TFLite-{quantization}"

    @classmethod
    def from_fitted(
        cls, model: SequenceModel, quantization: Quantization, calibration: RaceDataset | None
    ) -> TFLiteRaceModel:
        lite = cls(model, quantization)
        lite._convert(model, calibration)
        return lite

    def fit(self, ds: RaceDataset) -> TFLiteRaceModel:
        """Fit the source model on ``ds``, then convert it (``ds`` also calibrates int8)."""
        self._convert(self.source().fit(ds), ds)
        return self

    def _model_inputs(self, model: SequenceModel, ds: RaceDataset) -> list[np.ndarray]:
        if isinstance(model, CompactModel):
            return model.student_inputs(ds)
        return model.inputs(ds)

    def _convert(self, model: SequenceModel, calibration: RaceDataset | None) -> None:
        import tensorflow as tf

        self.fitted_ = model
        rep = self._model_inputs(model, calibration) if calibration is not None else None
        self.flatbuffer_ = to_tflite(model.model_, self.quantization, rep)
        self.interpreter_ = tf.lite.Interpreter(model_content=self.flatbuffer_)
        # Map our input order to the interpreter's by tensor name (seq_input, mask_input, ...).
        names = [t.name for t in model.model_.inputs]
        details = self.interpreter_.get_input_details()
        self.input_index_ = [
            next(d["index"] for d in details if name in d["name"]) for name in names
        ]
        self.output_index_ = self.interpreter_.get_output_details()[0]["index"]

    @property
    def size_kb(self) -> float:
        return len(self.flatbuffer_) / 1024

    def _invoke(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Run one batch, resizing the batch dimension if needed."""
        it = self.interpreter_
        n = len(inputs[0])
        resize = it.get_input_details()[0]["shape"][0] != n
        if resize:
            for idx, a in zip(self.input_index_, inputs, strict=True):
                it.resize_tensor_input(idx, a.shape)
        if resize or not getattr(self, "_allocated", False):
            it.allocate_tensors()
            self._allocated = True
        for idx, a in zip(self.input_index_, inputs, strict=True):
            it.set_tensor(idx, a.astype("float32"))
        it.invoke()
        return it.get_tensor(self.output_index_).reshape(-1)

    def predict(self, ds: RaceDataset) -> np.ndarray:
        pred = self._invoke(self._model_inputs(self.fitted_, ds))
        return self.fitted_.y_scaler_.inverse_transform(pred.reshape(-1, 1)).ravel()


def latency_ms(predict_one: Callable[[], object], repeats: int = 100, warmup: int = 5) -> float:
    """Median wall-clock latency of a single-example prediction, in milliseconds."""
    for _ in range(warmup):
        predict_one()
    times = []
    for _ in range(repeats):
        with timer() as t:
            predict_one()
        times.append(t.seconds)
    return float(np.median(times) * 1000)
