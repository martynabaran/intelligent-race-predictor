import numpy as np
import pandas as pd
import pytest

from race_predictor.features import build_race_dataset

pytest.importorskip("tensorflow")

from race_predictor.models.compact import (  # noqa: E402
    STANDARD_DISTANCES_M,
    CompactModel,
    DistilledModel,
    to_student_inputs,
    with_race_distance,
)
from race_predictor.models.sequence import PAD_VALUE, SequenceModel, build_model  # noqa: E402
from race_predictor.models.tflite import TFLiteRaceModel, latency_ms, to_tflite  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    from conftest import make_frame

    rows = []
    for a, speed in zip("abcd", (3.0, 3.4, 3.8, 4.2), strict=True):
        for i in range(14):
            rows.append(
                dict(
                    athlete=a,
                    date=pd.Timestamp("2023-01-01") + pd.Timedelta(days=7 * i),
                    avg_speed_ms=speed,
                    is_race=False,
                    is_triathlon=0,
                )
            )
        for j in range(3):
            rows.append(
                dict(
                    athlete=a,
                    date=pd.Timestamp("2023-04-20") + pd.Timedelta(days=14 * j),
                    distance_m=10000.0,
                    avg_speed_ms=speed * 1.1,
                    moving_s=10000 / (speed * 1.1),
                    is_race=True,
                    is_triathlon=0,
                )
            )
    return build_race_dataset(make_frame(rows))


@pytest.fixture(scope="module")
def teacher(ds):
    return SequenceModel("tcn", epochs=3, width=8).fit(ds)


@pytest.fixture(scope="module")
def distilled(ds, teacher):
    return DistilledModel(epochs=3, n_augment=2).fit(ds, teacher=teacher)


def test_student_inputs_replace_padding_with_mask():
    seq = np.array([[[1.0, 2.0], [PAD_VALUE, PAD_VALUE]]], dtype="float32")
    static = np.zeros((1, 3), dtype="float32")
    s, mask, st = to_student_inputs([seq, static])
    assert mask.tolist() == [[1.0, 0.0]]
    assert s[0, 1].tolist() == [0.0, 0.0] and s[0, 0].tolist() == [1.0, 2.0]
    assert st is static


def test_compact_is_much_smaller_than_default_teacher(ds):
    student = CompactModel(epochs=2).fit(ds)
    pred = student.predict(ds)
    assert pred.shape == (len(ds),) and np.isfinite(pred).all()
    seq_shape, static_shape = student.model_.inputs[0].shape[1:], student.model_.inputs[2].shape[1:]
    default_tcn = build_model("tcn", seq_shape, static_shape)
    assert student.model_.count_params() * 10 < default_tcn.count_params()


def test_compact_padding_does_not_change_prediction(ds):
    model = CompactModel(epochs=2).fit(ds)
    assert model.predict(ds.subset([0])) == pytest.approx(model.predict(ds)[:1], abs=1e-3)


def test_distilled_reuses_teacher_preprocessing(ds, teacher, distilled):
    assert distilled.teacher_ is teacher
    assert distilled.y_scaler_ is teacher.y_scaler_ and distilled.max_len_ == teacher.max_len_
    assert np.isfinite(distilled.predict(ds)).all()


def test_distilled_fits_its_own_teacher(ds):
    model = DistilledModel(
        teacher=lambda: SequenceModel("tcn", epochs=1, width=4), epochs=1, n_augment=0
    ).fit(ds)
    assert isinstance(model.teacher_, SequenceModel)


def test_distilled_rejects_bad_alpha():
    with pytest.raises(ValueError):
        DistilledModel(alpha=1.5)


def test_transfer_set_uses_standard_distances(ds, distilled):
    rng = np.random.default_rng(0)
    seq, static = distilled._transfer_set(ds, rng)
    assert len(seq) == len(static) == 2 * len(ds)
    dist = with_race_distance(ds, np.full(len(ds), 5000.0))
    assert (dist.static["race_distance_m"] == 5000.0).all() and dist.y is ds.y
    assert 10000.0 in STANDARD_DISTANCES_M


@pytest.mark.parametrize("quantization", ["float32", "dynamic", "int8"])
def test_tflite_matches_keras(ds, distilled, quantization):
    lite = TFLiteRaceModel.from_fitted(distilled, quantization, calibration=ds)
    keras_pred = distilled.predict(ds)
    lite_pred = lite.predict(ds)
    tol = {"float32": 1e-2, "dynamic": 2.0, "int8": 10.0}[quantization]  # s/km
    assert lite_pred == pytest.approx(keras_pred, abs=tol)
    assert lite.predict(ds.subset([0])) == pytest.approx(lite_pred[:1], abs=1e-3)
    assert lite.size_kb > 0


def test_tflite_teacher_float32(ds, teacher):
    lite = TFLiteRaceModel.from_fitted(teacher, "float32", calibration=None)
    assert lite.predict(ds) == pytest.approx(teacher.predict(ds), abs=1e-2)


def test_tflite_factory_interface(ds):
    lite = TFLiteRaceModel(lambda: CompactModel(epochs=1), "dynamic").fit(ds)
    assert lite.predict(ds).shape == (len(ds),)


def test_int8_needs_calibration(teacher):
    with pytest.raises(ValueError):
        to_tflite(teacher.model_, "int8")


def test_latency_ms_positive():
    assert latency_ms(lambda: sum(range(100)), repeats=5, warmup=1) >= 0
