"""Regression metrics. NRMSE is RMSE divided by the range of the true values."""

from __future__ import annotations

import numpy as np


def evaluate(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    span = float(y_true.max() - y_true.min()) or 1.0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "nrmse_pct": 100 * rmse / span,
        "mape_pct": float(100 * np.mean(np.abs(err) / y_true)),
    }
