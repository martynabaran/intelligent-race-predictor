"""TFLite export (float / full-integer int8), and an interpreter wrapper with benchmarking."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

INPUT_NAMES = ("seq_input", "mask_input", "static_input")


def unroll_rnns(model):
    """Copy of ``model`` with every RNN layer unrolled (same weights).

    A rolled GRU/LSTM converts to a TensorList while-loop that needs Select TF ops (the Flex
    delegate), which the stock TFLite interpreter - and a microcontroller - cannot run. Unrolled
    over the fixed sequence length it lowers to builtin ops, so it also quantises to int8.
    Models without RNN layers are returned unchanged.
    """
    import keras

    rnns = [layer for layer in model._flatten_layers() if isinstance(layer, keras.layers.RNN)]
    if not rnns:
        return model

    def clone(layer):
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
    model,
    mode: str = "float",
    representative: Sequence[np.ndarray] | None = None,
    select_tf_ops: bool = False,
    max_calibration: int = 100,
) -> bytes:
    """Convert a Keras model. ``mode``: ``float`` or ``int8`` (weights+activations quantised,
    float32 input/output kept so callers need no (de)quantisation code).

    ``representative``: list of input arrays (one per model input, first axis = samples), needed
    for ``int8`` calibration.

    RNN layers are unrolled first (:func:`unroll_rnns`), so ``select_tf_ops`` is only needed for
    other unsupported ops.
    """
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(unroll_rnns(model))
    if select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
    if mode == "int8":
        if representative is None:
            raise ValueError("int8 mode needs a representative dataset")
        n = min(len(representative[0]), max_calibration)

        def calibration():
            for i in range(n):
                yield [a[i : i + 1].astype("float32") for a in representative]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif mode != "float":
        raise ValueError("mode must be 'float' or 'int8'")
    return converter.convert()


class TFLiteModel:
    """Run a serialised TFLite model one sample at a time (batch size 1, as on a device)."""

    def __init__(self, content: bytes, num_threads: int = 1):
        import tensorflow as tf

        self.content = content
        self.interpreter = tf.lite.Interpreter(model_content=content, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()
        self._out = self.interpreter.get_output_details()[0]

    @property
    def size_kb(self) -> float:
        return len(self.content) / 1024

    def _order(self, arrays: Sequence[np.ndarray]) -> list[int]:
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
        order = self._order(arrays)
        n = len(arrays[0])
        return np.array([self._run_one([a[i : i + 1] for a in arrays], order) for i in range(n)])

    def latency_us(self, arrays: Sequence[np.ndarray], repeats: int = 200) -> float:
        """Median wall-clock microseconds of one inference (single thread, host CPU)."""
        order = self._order(arrays)
        sample = [a[:1] for a in arrays]
        for _ in range(10):
            self._run_one(sample, order)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self._run_one(sample, order)
            times.append((time.perf_counter() - t0) * 1e6)
        return float(np.median(times))
