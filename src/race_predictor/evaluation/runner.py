"""Run several models under both evaluation protocols and tabulate the results."""

from __future__ import annotations

import argparse
from collections.abc import Callable
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
from race_predictor.processing import default_pipeline
from race_predictor.sources import load_directory
from race_predictor.utils import timer

ModelFactory = Callable[[], RaceModel]


def _score(ds: RaceDataset, pred: np.ndarray, index: np.ndarray) -> dict[str, float]:
    sub = ds.subset(index)
    dist_km = sub.meta["race_distance_m"].to_numpy() / 1000
    time = evaluate(sub.y * dist_km, pred * dist_km)
    pace = evaluate(sub.y, pred)
    return {
        "MAE_time_min": time["mae"] / 60,
        "RMSE_time_min": time["rmse"] / 60,
        "MAPE_%": time["mape_pct"],
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
            train, test = chronological_split(ds)
            pred = make().fit(ds.subset(train)).predict(ds.subset(test))
            rows.append({"model": name, "protocol": "chronological", **_score(ds, pred, test)})

            oof, order = [], []
            for tr, te in grouped_splits(ds, n_splits):
                oof.append(make().fit(ds.subset(tr)).predict(ds.subset(te)))
                order.append(te)
            order = np.concatenate(order)
            rows.append(
                {"model": name, "protocol": "grouped-CV", **_score(ds, np.concatenate(oof), order)}
            )
        for r in rows[-2:]:
            r["fit_s"] = t.seconds
    return pd.DataFrame(rows)


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


def load_dataset(data_dir: Path) -> RaceDataset:
    frames = [
        load_directory(data_dir / "strava_activities"),
        load_directory(data_dir / "garmin_activities"),
    ]
    return build_race_dataset(default_pipeline().run(pd.concat(frames, ignore_index=True)))


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
