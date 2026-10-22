"""Run several models under both evaluation protocols and tabulate the results."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import chronological_split, grouped_splits
from race_predictor.features import RaceDataset, build_race_dataset
from race_predictor.models import (
    HistGradientBoostingBaseline,
    LightGBMBaseline,
    RidgeBaseline,
    RiegelBaseline,
)
from race_predictor.models.base import RaceModel
from race_predictor.processing import Pipeline, cleaning_pipeline, load_race_labels
from race_predictor.sources import load_directory
from race_predictor.utils import timer

ModelFactory = Callable[[], RaceModel]
LABELS_FILE = "race_labels.csv"


def _score(ds: RaceDataset, pred: np.ndarray, index: np.ndarray) -> dict[str, float]:
    sub = ds.subset(index)
    dist_km = sub.meta["race_distance_m"].to_numpy() / 1000
    time = evaluate(sub.y * dist_km, pred * dist_km)
    pace = evaluate(sub.y, pred)
    return {
        "MAE_time_min": time["mae"] / 60,
        "RMSE_time_min": time["rmse"] / 60,
        "MAPE_%": time["mape_pct"],
        "MedAPE_%": time["median_ape_pct"],
        "MAE_pace_s_km": pace["mae"],
    }


def compare_models(
    ds: RaceDataset, factories: dict[str, ModelFactory], n_splits: int = 5
) -> pd.DataFrame:
    """One row per (model, protocol). Grouped CV scores are computed on pooled out-of-fold
    predictions."""
    rows = []
    for name, make in factories.items():
        with timer(name) as t:
            for protocol, pred, index in protocol_predictions(ds, make, n_splits):
                rows.append({"model": name, "protocol": protocol, **_score(ds, pred, index)})
        for r in rows[-2:]:
            r["fit_s"] = t.seconds
    return pd.DataFrame(rows)


def protocol_predictions(
    ds: RaceDataset, make: ModelFactory, n_splits: int = 5
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(protocol, predictions, test_index)`` for both evaluation protocols. Grouped CV
    yields the pooled out-of-fold predictions."""
    train, test = chronological_split(ds)
    yield "chronological", make().fit(ds.subset(train)).predict(ds.subset(test)), test

    oof, order = [], []
    for tr, te in grouped_splits(ds, n_splits):
        oof.append(make().fit(ds.subset(tr)).predict(ds.subset(te)))
        order.append(te)
    yield "grouped-CV", np.concatenate(oof), np.concatenate(order)


def default_factories(with_gru: bool = False) -> dict[str, ModelFactory]:
    factories: dict[str, ModelFactory] = {
        "Riegel": RiegelBaseline,
        "Ridge": RidgeBaseline,
        "HistGB": HistGradientBoostingBaseline,
    }
    try:
        import lightgbm  # noqa: F401

        factories["LightGBM"] = LightGBMBaseline
    except (ImportError, OSError):
        print("LightGBM unavailable (on macOS: `brew install libomp`) - skipping")
    if with_gru:
        from race_predictor.models.gru import GRUModel

        factories["GRU"] = GRUModel
    return factories


def load_cleaned(data_dir: Path, pipeline: Pipeline | None = None) -> pd.DataFrame:
    """Cleaned activities of the Strava + Garmin exports in ``data_dir``.

    ``pipeline`` defaults to ``cleaning_pipeline("train")`` with the hand-labelled races of
    ``data_dir/race_labels.csv`` when that file exists (see ``processing.labels``).
    """
    frames = [
        load_directory(data_dir / "strava_activities"),
        load_directory(data_dir / "garmin_activities"),
    ]
    if pipeline is None:
        labels_path = data_dir / LABELS_FILE
        labels = load_race_labels(labels_path) if labels_path.exists() else None
        pipeline = cleaning_pipeline("train", race_labels=labels)
    return pipeline.run(pd.concat(frames, ignore_index=True))


def load_dataset(data_dir: Path, pipeline: Pipeline | None = None) -> RaceDataset:
    """Races from the Strava + Garmin exports (cleaning as in :func:`load_cleaned`)."""
    return build_race_dataset(load_cleaned(data_dir, pipeline))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_BASELINES.md"))
    ap.add_argument("--with-gru", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.data)
    print(f"{len(ds)} races from {ds.meta['athlete'].nunique()} athletes")
    table = compare_models(ds, default_factories(args.with_gru)).round(2)
    print(table.to_string(index=False))
    args.out.write_text(
        f"# Baseline results\n\n{len(ds)} races, {ds.meta['athlete'].nunique()} athletes. "
        "Metrics are on total race time (pace x distance). `chronological` = thesis protocol "
        "(latest 20% of each athlete's races held out); `grouped-CV` = whole athletes held out.\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
