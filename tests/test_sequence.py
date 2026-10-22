import numpy as np
import pandas as pd
import pytest

from race_predictor.features import build_race_dataset

pytest.importorskip("tensorflow")

from race_predictor.evaluation.explain import (
    attribution_by_channel,
    attribution_by_period,
    integrated_gradients,
    shap_tabular,
)
from race_predictor.models import HistGradientBoostingBaseline
from race_predictor.models.sequence import SequenceModel


@pytest.fixture
def ds(frame):
    rows = []
    for a, speed in zip("abc", (3.0, 3.5, 4.0), strict=True):
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


@pytest.mark.parametrize("encoder", ["gru", "tcn", "transformer"])
def test_encoders_fit_predict(ds, encoder):
    model = SequenceModel(encoder, epochs=2, width=8).fit(ds)
    pred = model.predict(ds)
    assert pred.shape == (len(ds),) and np.isfinite(pred).all()


@pytest.mark.parametrize("encoder", ["gru", "tcn", "transformer"])
def test_padding_does_not_change_prediction(ds, encoder):
    """Prediction for a race must not depend on how much padding its batch needs."""
    model = SequenceModel(encoder, epochs=2, width=8).fit(ds)
    alone = model.predict(ds.subset([0]))
    together = model.predict(ds)[:1]
    assert alone == pytest.approx(together, abs=1e-3)


def test_unknown_encoder():
    with pytest.raises(ValueError):
        SequenceModel("lstm")


def test_integrated_gradients_completeness(ds):
    """sum(attributions) ~= f(x) - f(baseline) (the defining IG property)."""
    import tensorflow as tf

    model = SequenceModel("transformer", epochs=2, width=8).fit(ds)
    sub = ds.subset(range(4))
    seq_attr, static_attr = integrated_gradients(model, sub, steps=256)
    total = np.array([a.sum() for a in seq_attr]) + static_attr.sum(axis=1)

    seq, static = (tf.constant(a) for a in model.inputs(sub))
    pad = tf.cast(seq[:, :, :1] == 999.0, tf.float32)
    base = model.model_([seq * pad, tf.zeros_like(static)], training=False).numpy().ravel()
    full = model.model_([seq, static], training=False).numpy().ravel()
    expected = (full - base) * model.y_scaler_.scale_[0]
    assert total == pytest.approx(expected, rel=0.15, abs=6.0)

    assert len(seq_attr) == 4 and attribution_by_period(sub, seq_attr).shape == (13, 3)
    assert len(attribution_by_channel(seq_attr)) == 11


def test_shap_tabular_shape(ds):
    gb = HistGradientBoostingBaseline().fit(ds)
    values = shap_tabular(gb, ds)
    assert values.shape == (len(ds), len(ds.tabular_columns))
