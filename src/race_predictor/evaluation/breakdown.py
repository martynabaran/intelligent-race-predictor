"""Where do the errors come from? Scores per segment of races.

Triathlon runs (after a swim and a ride) are much slower and harder to predict than stand-alone
races of the same distance, and the app only predicts stand-alone races (``build_query_dataset``
sets ``is_triathlon = 0``). Pooled metrics therefore overstate the error that matters for the app;
this module reports stand-alone races, triathlon runs and each race distance separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from race_predictor.evaluation.metrics import evaluate
from race_predictor.features import RaceDataset
from race_predictor.processing import distance_label


def segment_masks(ds: RaceDataset) -> dict[str, np.ndarray]:
    """Boolean mask per segment, over all races of ``ds``."""
    tri = ds.static["is_triathlon"].to_numpy() == 1
    labels = ds.meta["race_distance_m"].map(distance_label).to_numpy()
    masks = {"all": np.ones(len(ds), dtype=bool), "stand-alone": ~tri, "triathlon": tri}
    for label in dict.fromkeys(labels):
        masks[f"{label} (stand-alone)"] = (labels == label) & ~tri
    return masks


def breakdown(ds: RaceDataset, pred: np.ndarray, index: np.ndarray) -> pd.DataFrame:
    """One row per segment with at least one race in ``index`` (metrics on total race time)."""
    km = ds.meta["race_distance_m"].to_numpy()[index] / 1000
    true_t, pred_t = ds.y[index] * km, np.asarray(pred) * km
    rows = []
    for segment, mask in segment_masks(ds).items():
        sel = mask[index]
        if not sel.any():
            continue
        m = evaluate(true_t[sel], pred_t[sel])
        rows.append(
            {
                "segment": segment,
                "n": int(sel.sum()),
                "MAPE_%": m["mape_pct"],
                "MedAPE_%": m["median_ape_pct"],
                "MAE_time_min": m["mae"] / 60,
            }
        )
    return pd.DataFrame(rows)
