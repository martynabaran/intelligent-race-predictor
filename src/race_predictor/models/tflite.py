"""TFLite export (float / full-integer int8), and an interpreter wrapper with benchmarking."""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import Any

import keras
import numpy as np

# Keras input names in model order; the converter keeps them in the interpreter's input names.
INPUT_NAMES = ("seq_input", "mask_input", "static_input")


def unroll_rnns(model: keras.Model) -> keras.Model:
    """Copy of ``model`` with every RNN layer unrolled (same weights).

    A rolled GRU/LSTM converts to a TensorList while-loop that needs Select TF ops (the Flex
    delegate), which the stock TFLite interpreter - and a microcontroller - cannot run. Unrolled
    over the fixed sequence length it lowers to builtin ops, so it also quantises to int8.
    Models without RNN layers are returned unchanged.
    """
    rnns = [layer for layer in model._flatten_layers() if isinstance(layer, keras.layers.RNN)]
    if not rnns:
        return model

    def clone(layer: keras.layers.Layer) -> keras.layers.Layer:
        cfg = layer.get_config()
        if isinstance(layer, keras.layers.Bidirectional):
            cfg["layer"]["config"]["unroll"] = True
            if cfg.get("backward_layer"):
                cfg["backward_layer"]["config"]["unroll"] = True
        elif isinstance(layer, keras.layers.RNN):
            cfg["unroll"] = True
        return layer.__class__.from_config(cfg)

    unrolled = keras.models.clone_model(model, clone_function=clone)
    unrolled.set_weights(model.get_weights())
    return unrolled


def convert_to_tflite(
    model: keras.Model,
    mode: str = "float",
    representative: Sequence[np.ndarray] | None = None,
    *,
    select_tf_ops: bool = False,
    max_calibration: int = 100,
) -> bytes:
    """Convert a Keras model to a serialised TFLite flatbuffer.

    Args:
        model: Model to convert; RNN layers are unrolled first (:func:`unroll_rnns`), so
            ``select_tf_ops`` is only needed for other unsupported ops.
        mode: ``float`` or ``int8`` (weights and activations quantised, float32 input/output kept
            so callers need no (de)quantisation code).
        representative: Input arrays (one per model input, first axis = samples), needed for
            ``int8`` calibration.
        select_tf_ops: Allow Select TF ops (needs the Flex delegate to run).
        max_calibration: Number of ``representative`` samples used for calibration.

    Returns:
        The TFLite model bytes.

    Raises:
        ValueError: If ``mode`` is unknown or ``int8`` has no ``representative`` data.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(unroll_rnns(model))
    if select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        # Private converter flag with no public equivalent: keep TensorList ops as Select TF ops
        # instead of lowering them to builtins, which fails for loops that cannot be lowered.
        converter._experimental_lower_tensor_list_ops = False
    if mode == "int8":
        if representative is None:
            raise ValueError("int8 mode needs a representative dataset")
        n_samples = min(len(representative[0]), max_calibration)

        def calibration() -> Iterator[list[np.ndarray]]:
            for i in range(n_samples):
                yield [arr[i : i + 1].astype("float32") for arr in representative]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif mode != "float":
        raise ValueError("mode must be 'float' or 'int8'")
    return converter.convert()


class TFLiteModel:
    """Run a serialised TFLite model one sample at a time (batch size 1, as on a device)."""

    def __init__(self, content: bytes, num_threads: int = 1) -> None:
        import tensorflow as tf

        self.content = content
        self.interpreter = tf.lite.Interpreter(model_content=content, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self._in: list[dict[str, Any]] = self.interpreter.get_input_details()
        self._out: dict[str, Any] = self.interpreter.get_output_details()[0]

    @property
    def size_kb(self) -> float:
        """Size of the serialised model in KiB."""
        return len(self.content) / 1024

    def _order(self) -> list[int]:
        """Map each Keras input (by position) to the interpreter input index (matched by name)."""
        if len(self._in) == 1:
            return [0]
        names = [d["name"] for d in self._in]
        keys = [k for k in INPUT_NAMES if any(k in n for n in names)]
        return [next(i for i, n in enumerate(names) if key in n) for key in keys]

    def _run_one(self, sample: Sequence[np.ndarray], order: list[int]) -> float:
        for arr, idx in zip(sample, order, strict=True):
            self.interpreter.set_tensor(self._in[idx]["index"], arr.astype("float32"))
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self._out["index"]).ravel()[0])

    def predict(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        """Predict every sample of ``arrays`` (one array per model input, first axis = samples)."""
        order = self._order()
        n_samples = len(arrays[0])
        return np.array(
            [self._run_one([arr[i : i + 1] for arr in arrays], order) for i in range(n_samples)]
        )

    def latency_us(self, arrays: Sequence[np.ndarray], repeats: int = 200) -> float:
        """Median wall-clock microseconds of one inference (single thread, host CPU)."""
        order = self._order()
        sample = [arr[:1] for arr in arrays]
        for _ in range(10):
            self._run_one(sample, order)
        durations_us = []
        for _ in range(repeats):
            start = time.perf_counter()
            self._run_one(sample, order)
            durations_us.append((time.perf_counter() - start) * 1e6)
        return float(np.median(durations_us))
