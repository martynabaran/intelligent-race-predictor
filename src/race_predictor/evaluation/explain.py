"""Explainability: SHAP for tabular models, integrated gradients for sequence models.

Attributions are in seconds-per-km of predicted race pace (positive = predicted slower).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from race_predictor.features import SEQ_FEATURES, STATIC_BINARY, STATIC_NUMERIC, RaceDataset
from race_predictor.models.baselines import _TabularModel


def shap_tabular(
    model: _TabularModel, ds: RaceDataset, max_evals: int | None = None
) -> pd.DataFrame:
    """Per-race SHAP values (model-agnostic permutation explainer) for a fitted tabular model."""
    import shap

    x = model._x(ds)
    cols = ds.tabular_columns
    explainer = shap.Explainer(
        model.model_.predict, shap.maskers.Independent(x, max_samples=50), algorithm="permutation"
    )
    values = explainer(x, max_evals=max_evals or 2 * len(cols) + 1).values
    return pd.DataFrame(values, columns=cols)


def integrated_gradients(model, ds: RaceDataset, steps: int = 256):
    """Integrated gradients for a fitted :class:`SequenceModel`.

    Baseline: the training mean (zero in standardised space) for every real step / static
    feature; padded steps are left untouched so they receive no attribution.

    LayerNorm-based encoders (TCN, Transformer) need many steps to satisfy completeness
    (sum of attributions = f(x) - f(baseline)); 256 gives errors of a few s/km.

    Returns ``(seq_attr, static_attr)``: a list of ``(n_steps_i, n_features)`` arrays (aligned to
    ``ds.seqs``, oldest step first) and an ``(n, n_static)`` array, both in s/km of pace.
    """
    import tensorflow as tf

    seq, static = (tf.constant(a) for a in model.inputs(ds))
    pad = tf.cast(seq[:, :, :1] == 999.0, tf.float32)
    seq_base = seq * pad  # real steps -> 0, pad steps keep the pad value
    static_base = tf.zeros_like(static)

    seq_grad = tf.zeros_like(seq)
    static_grad = tf.zeros_like(static)
    for alpha in (np.arange(steps) + 0.5) / steps:
        s = seq_base + alpha * (seq - seq_base)
        t = static_base + alpha * (static - static_base)
        with tf.GradientTape() as tape:
            tape.watch([s, t])
            out = model.model_([s, t], training=False)
        gs, gt = tape.gradient(out, [s, t])
        seq_grad += gs
        static_grad += gt
    scale = float(model.y_scaler_.scale_[0])
    seq_attr = (seq - seq_base) * seq_grad / steps * scale
    static_attr = (static - static_base) * static_grad / steps * scale

    seq_attr = seq_attr.numpy()
    out_seq = []
    for i, original in enumerate(ds.seqs):
        n = min(len(original), seq_attr.shape[1])
        out_seq.append(seq_attr[i, :n])  # truncating='pre' keeps the most recent n steps
    return out_seq, static_attr.numpy()


def static_feature_names() -> list[str]:
    return STATIC_NUMERIC + STATIC_BINARY


def attribution_by_period(
    ds: RaceDataset, seq_attr: list[np.ndarray], period_days: int = 28, n_periods: int = 13
) -> pd.DataFrame:
    """Net effect of each period before the race on the predicted pace (s/km).

    Attributions are summed (signed) over the steps and channels of a period *within each race*,
    then the absolute value is averaged over races, so opposite-signed contributions inside a
    period cancel instead of inflating the importance. Period 0 is the ``period_days`` just
    before the race. ``mean_signed`` shows the average direction (+ = slower).
    """
    day_col = SEQ_FEATURES.index("days_to_race")
    net = np.zeros((len(ds), n_periods))
    for i, (seq, attr) in enumerate(zip(ds.seqs, seq_attr, strict=True)):
        days = seq[-len(attr) :, day_col]
        bucket = np.minimum((days // period_days).astype(int), n_periods - 1)
        np.add.at(net[i], bucket, attr.sum(axis=1))
    return pd.DataFrame(
        {
            "period": [f"{i * period_days}-{(i + 1) * period_days} d" for i in range(n_periods)],
            "mean_abs_net_s_km": np.abs(net).mean(axis=0),
            "mean_signed_s_km": net.mean(axis=0),
        }
    )


def attribution_by_channel(seq_attr: list[np.ndarray]) -> pd.Series:
    """Mean over races of |net attribution| of each run feature (summed over its time steps)."""
    net = np.stack([a.sum(axis=0) for a in seq_attr])
    return pd.Series(np.abs(net).mean(axis=0), index=SEQ_FEATURES).sort_values(ascending=False)
