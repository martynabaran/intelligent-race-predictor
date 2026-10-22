"""Accuracy / size / latency of distilled, quantised models for on-device prediction.

For every training fold the teacher is fitted once (shared through a cache); each student is
trained plainly (``alpha = 1``) and distilled, then evaluated in every export format. Size and
latency are measured on models fitted on all data. Latency is single-sample, single-thread on the
host CPU - a relative indicator, not a watch measurement.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.evaluation.runner import _score
from race_predictor.evaluation.splits import chronological_split, grouped_splits
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.distill import Distilled
from race_predictor.models.tflite import TFLiteModel, convert_to_tflite


@dataclass
class StudentSpec:
    name: str
    factory: Callable[[], RaceModel]
    modes: tuple[str, ...] = ("float", "int8")


def _format_predictions(student, ds_train: RaceDataset, ds_test: RaceDataset, modes) -> dict:
    """Predictions (pace s/km) of a fitted student in Keras and each TFLite format."""
    out = {"keras": student.predict(ds_test)}
    for mode in modes:
        blob = convert_to_tflite(student.model_, mode, student.inputs(ds_train))
        out[f"tflite-{mode}"] = student.to_pace(TFLiteModel(blob).predict(student.inputs(ds_test)))
    return out


def evaluate_embedded(
    ds: RaceDataset,
    teacher_factory: Callable[[], RaceModel],
    students: list[StudentSpec],
    alpha: float = 0.5,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Accuracy per (student, distillation, format, protocol)."""
    cache: dict = {}
    folds = {
        "chronological": [chronological_split(ds)],
        "grouped-CV": list(grouped_splits(ds, n_splits)),
    }
    rows = []
    for spec in students:
        for a, label in ((1.0, "plain"), (alpha, f"distilled a={alpha}")):
            pooled: dict[str, dict[str, list]] = {p: {} for p in folds}
            for protocol, splits in folds.items():
                for tr, te in splits:
                    model = (
                        Distilled(teacher_factory, spec.factory, a, cache)
                        if a < 1
                        else spec.factory()
                    )
                    model.fit(ds.subset(tr))
                    student = model.student_ if a < 1 else model
                    preds = _format_predictions(student, ds.subset(tr), ds.subset(te), spec.modes)
                    for fmt, p in preds.items():
                        bucket = pooled[protocol].setdefault(fmt, [[], []])
                        bucket[0].append(p)
                        bucket[1].append(te)
            for protocol, formats in pooled.items():
                for fmt, (p, idx) in formats.items():
                    rows.append(
                        {
                            "student": spec.name,
                            "training": label,
                            "format": fmt,
                            "protocol": protocol,
                            **_score(ds, np.concatenate(p), np.concatenate(idx)),
                        }
                    )
    return pd.DataFrame(rows)


def teacher_scores(ds: RaceDataset, teacher_factory, n_splits: int = 5) -> pd.DataFrame:
    rows = []
    for protocol, splits in {
        "chronological": [chronological_split(ds)],
        "grouped-CV": list(grouped_splits(ds, n_splits)),
    }.items():
        p, idx = [], []
        for tr, te in splits:
            p.append(teacher_factory().fit(ds.subset(tr)).predict(ds.subset(te)))
            idx.append(te)
        rows.append(
            {
                "student": "teacher",
                "training": "-",
                "format": "keras",
                "protocol": protocol,
                **_score(ds, np.concatenate(p), np.concatenate(idx)),
            }
        )
    return pd.DataFrame(rows)


def measure_deployment(
    ds: RaceDataset, teacher_factory, students: list[StudentSpec], alpha: float = 0.5
) -> pd.DataFrame:
    """Size (KB) and latency (us) of each exported format, models fitted on all data."""
    rows = []
    teacher = teacher_factory().fit(ds)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "teacher.keras"
        teacher.model_.save(path)
        rows.append(
            {
                "model": "teacher",
                "format": "keras",
                "size_kb": path.stat().st_size / 1024,
                "latency_us": np.nan,
            }
        )
    x = teacher.inputs(ds)
    t0 = _time_keras(teacher.model_, x)
    rows[-1]["latency_us"] = t0
    for mode in ("float", "int8"):
        try:
            blob = convert_to_tflite(teacher.model_, mode, x)
            rows.append(
                {
                    "model": "teacher",
                    "format": f"tflite-{mode}",
                    "size_kb": len(blob) / 1024,
                    "latency_us": TFLiteModel(blob).latency_us(x, repeats=30),
                }
            )
        except Exception as e:  # conversion limits are a finding, not a crash
            rows.append(
                {
                    "model": "teacher",
                    "format": f"tflite-{mode}",
                    "size_kb": np.nan,
                    "latency_us": np.nan,
                    "note": f"conversion failed: {type(e).__name__}",
                }
            )

    for spec in students:
        for a, label in ((1.0, "plain"), (alpha, "distilled")):
            student = spec.factory().fit(ds.with_targets(a * ds.y + (1 - a) * teacher.predict(ds)))
            for mode in spec.modes:
                blob = convert_to_tflite(student.model_, mode, student.inputs(ds))
                m = TFLiteModel(blob)
                rows.append(
                    {
                        "model": f"{spec.name} {label}",
                        "format": f"tflite-{mode}",
                        "size_kb": m.size_kb,
                        "latency_us": m.latency_us(student.inputs(ds)),
                    }
                )
    return pd.DataFrame(rows)


def _time_keras(model, inputs, repeats: int = 30) -> float:
    import time

    sample = [a[:1] for a in inputs]
    model.predict(sample, verbose=0)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(sample, training=False)
        times.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(times))
