import numpy as np
import pandas as pd
import pytest

from race_predictor.features import build_race_dataset

pytest.importorskip("tensorflow")

from race_predictor.evaluation.embedded import StudentSpec, evaluate_embedded
from race_predictor.models.base import RaceModel
from race_predictor.models.distill import CompactSequence, Distilled, TabularMLP
from race_predictor.models.sequence import SequenceModel
from race_predictor.models.tflite import (
    TFLiteModel,
    convert_to_tflite,
    unroll_rnns,
)


@pytest.fixture
def ds(frame):
    rows = []
    for a, speed in zip("abcd", (3.0, 3.4, 3.8, 4.2), strict=True):
        for i in range(14):
            rows.append(
                {
                    "athlete": a,
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=7 * i),
                    "avg_speed_ms": speed,
                    "is_race": False,
                    "is_triathlon": 0,
                }
            )
        for j in range(3):
            rows.append(
                {
                    "athlete": a,
                    "date": pd.Timestamp("2023-04-20") + pd.Timedelta(days=14 * j),
                    "distance_m": 10000.0,
                    "avg_speed_ms": speed * 1.1,
                    "moving_s": 10000 / (speed * 1.1),
                    "is_race": True,
                    "is_triathlon": 0,
                }
            )
    return build_race_dataset(frame(rows))


class ConstTeacher(RaceModel):
    """Teacher that always predicts 250 s/km; counts how often it is fitted."""

    fits = 0

    def fit(self, ds):
        ConstTeacher.fits += 1
        return self

    def predict(self, ds):
        return np.full(len(ds), 250.0)


def test_distilled_targets_blend_teacher_and_truth(ds):
    seen = {}

    class Spy(RaceModel):
        def fit(self, d):
            seen["y"] = d.y.copy()
            return self

        def predict(self, d):
            return np.zeros(len(d))

    Distilled(ConstTeacher, Spy, alpha=0.25).fit(ds)
    assert seen["y"] == pytest.approx(0.25 * ds.y + 0.75 * 250.0)


def test_teacher_cache_reuses_fit(ds):
    ConstTeacher.fits = 0
    cache: dict = {}
    for _ in range(3):
        Distilled(ConstTeacher, TabularMLP, 0.5, cache).teacher(ds)
    assert ConstTeacher.fits == 1


@pytest.mark.parametrize("mode", ["float", "int8"])
def test_tflite_matches_keras(ds, mode):
    mlp = TabularMLP(epochs=3).fit(ds)
    x = mlp.inputs(ds)
    tfl = TFLiteModel(convert_to_tflite(mlp.model_, mode, x))
    ref = mlp.model_.predict(x[0], verbose=0).ravel()
    tol = 1e-4 if mode == "float" else 1.5  # int8 error in standardised-target units
    assert np.abs(tfl.predict(x) - ref).max() < tol
    assert tfl.size_kb > 0 and tfl.latency_us(x, repeats=5) > 0


def test_int8_needs_representative_data(ds):
    mlp = TabularMLP(epochs=1).fit(ds)
    with pytest.raises(ValueError):
        convert_to_tflite(mlp.model_, "int8")


def test_evaluate_embedded_table(ds):
    table = evaluate_embedded(
        ds, ConstTeacher, [StudentSpec("mlp", lambda: TabularMLP(epochs=2))], n_splits=2
    )
    assert set(table["format"]) == {"keras", "tflite-float", "tflite-int8"}
    assert set(table["training"]) == {"plain", "distilled a=0.5"}
    assert set(table["protocol"]) == {"chronological", "grouped-CV"}


@pytest.mark.parametrize("mode", ["float", "int8"])
def test_compact_sequence_tflite_matches_keras(ds, mode):
    student = CompactSequence(epochs=3).fit(ds)
    x = student.inputs(ds)
    assert [a.shape[0] for a in x] == [len(ds)] * 3 and (x[0][x[1] == 0] == 0).all()
    tfl = TFLiteModel(convert_to_tflite(student.model_, mode, x))
    ref = student.model_.predict(x, verbose=0).ravel()
    tol = 1e-4 if mode == "float" else 0.2  # standardised-target units
    assert np.abs(tfl.predict(x) - ref).max() < tol


def test_compact_sequence_padding_invariant(ds):
    student = CompactSequence(epochs=2).fit(ds)
    assert student.predict(ds.subset([0])) == pytest.approx(student.predict(ds)[:1], abs=1e-3)


def test_gru_teacher_converts_to_builtin_ops(ds):
    """Unrolled GRU runs on the stock interpreter (no Flex delegate), also as int8."""
    teacher = SequenceModel("gru", epochs=1, width=4).fit(ds)
    x = teacher.inputs(ds)
    unrolled = unroll_rnns(teacher.model_)
    full = (x[0][:, :, 0] != 999.0).all(axis=1)  # unpadded rows: unrolling is exact there
    ref = teacher.model_.predict(x, verbose=0).ravel()
    assert unrolled.predict(x, verbose=0).ravel()[full] == pytest.approx(ref[full], abs=1e-5)
    for mode in ("float", "int8"):
        assert np.isfinite(TFLiteModel(convert_to_tflite(teacher.model_, mode, x)).predict(x)).all()


def test_unroll_rnns_is_noop_without_rnn(ds):
    mlp = TabularMLP(epochs=1).fit(ds)
    assert unroll_rnns(mlp.model_) is mlp.model_
