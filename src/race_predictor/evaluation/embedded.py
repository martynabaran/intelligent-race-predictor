"""Accuracy / size / latency of distilled, quantised models for on-device prediction.

For every training fold the teacher is fitted once (shared through a cache); each student is
trained plainly (``alpha = 1``) and distilled, then evaluated in every export format. Size and
latency are measured on models fitted on all data. Latency is single-sample, single-thread on the
host CPU - a relative indicator, not a watch measurement.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from race_predictor.evaluation.runner import _score
from race_predictor.evaluation.splits import chronological_split, grouped_splits
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.distill import Distilled
from race_predictor.models.tflite import TFLiteModel, convert_to_tflite

Folds = dict[str, list[tuple[np.ndarray, np.ndarray]]]
# protocol -> export format -> (predictions per test fold, test indices per fold)
PooledPredictions = dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray]]]]


@dataclass
class StudentSpec:
    """A student architecture to evaluate.

    Attributes:
        name: Label used in result tables.
        factory: Builds a fresh, unfitted student.
        modes: TFLite quantisation modes to evaluate besides the Keras model.
    """

    name: str
    factory: Callable[[], RaceModel]
    modes: tuple[str, ...] = ("float", "int8")


def _format_predictions(
    student: Any, ds_train: RaceDataset, ds_test: RaceDataset, modes: Sequence[str]
) -> dict[str, np.ndarray]:
    """Predictions (pace s/km) of a fitted student in Keras and each TFLite format."""
    out = {"keras": student.predict(ds_test)}
    for mode in modes:
        blob = convert_to_tflite(student.model_, mode, student.inputs(ds_train))
        out[f"tflite-{mode}"] = student.to_pace(TFLiteModel(blob).predict(student.inputs(ds_test)))
    return out


def _protocol_folds(ds: RaceDataset, n_splits: int) -> Folds:
    """Train/test index pairs of both evaluation protocols."""
    return {
        "chronological": [chronological_split(ds)],
        "grouped-CV": list(grouped_splits(ds, n_splits)),
    }


def _pool_student_predictions(
    ds: RaceDataset,
    folds: Folds,
    spec: StudentSpec,
    teacher_factory: Callable[[], RaceModel],
    student_alpha: float,
    cache: dict,
) -> PooledPredictions:
    """Fit the student on every training fold; collect its per-format test predictions.

    ``student_alpha`` below 1 distils from the teacher, otherwise the student trains plainly.
    """
    pooled: PooledPredictions = {protocol: {} for protocol in folds}
    for protocol, splits in folds.items():
        for train_idx, test_idx in splits:
            ds_train = ds.subset(train_idx)
            if student_alpha < 1:
                distilled = Distilled(teacher_factory, spec.factory, student_alpha, cache)
                distilled.fit(ds_train)
                student = distilled.student_
            else:
                student = spec.factory()
                student.fit(ds_train)
            preds = _format_predictions(student, ds_train, ds.subset(test_idx), spec.modes)
            for fmt, pred in preds.items():
                bucket = pooled[protocol].setdefault(fmt, ([], []))
                bucket[0].append(pred)
                bucket[1].append(test_idx)
    return pooled


def evaluate_embedded(
    ds: RaceDataset,
    teacher_factory: Callable[[], RaceModel],
    students: list[StudentSpec],
    alpha: float = 0.5,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Accuracy per (student, distillation, format, protocol)."""
    cache: dict = {}
    folds = _protocol_folds(ds, n_splits)
    rows = []
    for spec in students:
        for student_alpha, label in ((1.0, "plain"), (alpha, f"distilled a={alpha}")):
            pooled = _pool_student_predictions(
                ds, folds, spec, teacher_factory, student_alpha, cache
            )
            for protocol, formats in pooled.items():
                for fmt, (preds, indices) in formats.items():
                    rows.append(
                        {
                            "student": spec.name,
                            "training": label,
                            "format": fmt,
                            "protocol": protocol,
                            **_score(ds, np.concatenate(preds), np.concatenate(indices)),
                        }
                    )
    return pd.DataFrame(rows)


def teacher_scores(
    ds: RaceDataset, teacher_factory: Callable[[], RaceModel], n_splits: int = 5
) -> pd.DataFrame:
    """Accuracy of the undistilled teacher (Keras format) under both protocols."""
    rows = []
    for protocol, splits in _protocol_folds(ds, n_splits).items():
        preds, indices = [], []
        for train_idx, test_idx in splits:
            preds.append(teacher_factory().fit(ds.subset(train_idx)).predict(ds.subset(test_idx)))
            indices.append(test_idx)
        rows.append(
            {
                "student": "teacher",
                "training": "-",
                "format": "keras",
                "protocol": protocol,
                **_score(ds, np.concatenate(preds), np.concatenate(indices)),
            }
        )
    return pd.DataFrame(rows)


def measure_deployment(
    ds: RaceDataset,
    teacher_factory: Callable[[], Any],
    students: list[StudentSpec],
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Size (KB) and latency (us) of each exported format, models fitted on all data.

    ``teacher_factory`` must build a Keras-backed model (with ``model_`` and ``inputs``).
    """
    rows: list[dict[str, object]] = []
    teacher = teacher_factory().fit(ds)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teacher.keras"
        teacher.model_.save(path)
        keras_size_kb = path.stat().st_size / 1024
    inputs = teacher.inputs(ds)
    rows.append(
        {
            "model": "teacher",
            "format": "keras",
            "size_kb": keras_size_kb,
            "latency_us": _time_keras(teacher.model_, inputs),
        }
    )
    for mode in ("float", "int8"):
        try:
            blob = convert_to_tflite(teacher.model_, mode, inputs)
            rows.append(
                {
                    "model": "teacher",
                    "format": f"tflite-{mode}",
                    "size_kb": len(blob) / 1024,
                    "latency_us": TFLiteModel(blob).latency_us(inputs, repeats=30),
                }
            )
        except (RuntimeError, ValueError) as error:  # converter limits are a finding, not a crash
            rows.append(
                {
                    "model": "teacher",
                    "format": f"tflite-{mode}",
                    "size_kb": np.nan,
                    "latency_us": np.nan,
                    "note": f"conversion failed: {type(error).__name__}",
                }
            )

    for spec in students:
        for student_alpha, label in ((1.0, "plain"), (alpha, "distilled")):
            blended = student_alpha * ds.y + (1 - student_alpha) * teacher.predict(ds)
            student: Any = spec.factory().fit(ds.with_targets(blended))
            for mode in spec.modes:
                blob = convert_to_tflite(student.model_, mode, student.inputs(ds))
                tflite_model = TFLiteModel(blob)
                rows.append(
                    {
                        "model": f"{spec.name} {label}",
                        "format": f"tflite-{mode}",
                        "size_kb": tflite_model.size_kb,
                        "latency_us": tflite_model.latency_us(student.inputs(ds)),
                    }
                )
    return pd.DataFrame(rows)


def _time_keras(model: Any, inputs: Sequence[np.ndarray], repeats: int = 30) -> float:
    """Median single-sample latency (us) of a Keras model's forward pass."""
    sample = [array[:1] for array in inputs]
    model.predict(sample, verbose=0)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(sample, training=False)
        times.append((time.perf_counter() - start) * 1e6)
    return float(np.median(times))
