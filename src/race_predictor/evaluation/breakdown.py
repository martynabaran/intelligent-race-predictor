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
    triathlon = ds.static["is_triathlon"].to_numpy() == 1
    labels = ds.meta["race_distance_m"].map(distance_label).to_numpy()
    masks = {"all": np.ones(len(ds), dtype=bool), "stand-alone": ~triathlon, "triathlon": triathlon}
    for label in dict.fromkeys(labels):
        masks[f"{label} (stand-alone)"] = (labels == label) & ~triathlon
    if "riegel_ref_is_race" in ds.static:  # what Riegel projects from: a race or a training run
        reference = ds.static["riegel_ref_is_race"].to_numpy()
        masks["reference: race"] = reference == 1
        masks["reference: training run"] = reference == 0
    return masks


def breakdown(ds: RaceDataset, pred: np.ndarray, index: np.ndarray) -> pd.DataFrame:
    """One row per segment with at least one race in ``index`` (metrics on total race time)."""
    km = ds.meta["race_distance_m"].to_numpy()[index] / 1000
    true_time, pred_time = ds.y[index] * km, np.asarray(pred) * km
    rows = []
    for segment, mask in segment_masks(ds).items():
        selected = mask[index]
        if not selected.any():
            continue
        metrics = evaluate(true_time[selected], pred_time[selected])
        rows.append(
            {
                "segment": segment,
                "n": int(selected.sum()),
                "MAPE_%": metrics["mape_pct"],
                "MedAPE_%": metrics["median_ape_pct"],
                "MAE_time_min": metrics["mae"] / 60,
            }
        )
    return pd.DataFrame(rows)
