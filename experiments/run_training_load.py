"""Stage 9c: do training-load features help, and how well does the Banister model fit?

    python experiments/run_training_load.py

Writes docs/RESULTS_TRAINING_LOAD.md. All numbers come from this script; the *Conclusions*
section is edited by hand after reading them (and kept when the script is rerun).
"""

import argparse
from pathlib import Path

import pandas as pd

from race_predictor.evaluation.breakdown import segment_masks
from race_predictor.evaluation.report import write_report
from race_predictor.evaluation.runner import load_cleaned
from race_predictor.evaluation.training_load import (
    banister_report,
    load_feature_comparison,
    paired_ape_difference,
)
from race_predictor.features import build_race_dataset
from race_predictor.models import (
    HistGradientBoostingBaseline,
    ModeRouter,
    RidgeBaseline,
    RiegelBaseline,
)
from race_predictor.training_load import LOAD_FEATURES

SEGMENTS = ["all", "stand-alone", "triathlon"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_TRAINING_LOAD.md"))
    args = ap.parse_args()

    cleaned = load_cleaned(args.data)
    ds = build_race_dataset(cleaned)
    print(f"{len(ds)} races, {ds.meta['athlete'].nunique()} athletes", flush=True)

    factories = {
        "Riegel (ignores features)": RiegelBaseline,
        "Ridge": RidgeBaseline,
        "HistGB": HistGradientBoostingBaseline,
        "Router (app model)": ModeRouter,
    }
    long, predictions = load_feature_comparison(ds, factories)

    # With running races only there may be no triathlon runs at all: use the segments present.
    masks = segment_masks(ds)
    segments = [s for s in SEGMENTS if masks[s].any()]
    scores = long[long["segment"].isin(segments)].pivot_table(
        index=["protocol", "model", "load"], columns="segment", values="MAPE_%", sort=False
    )[segments]
    scores.columns = [f"MAPE_% {s}" for s in segments]

    rows = []
    for (model, protocol), (without, with_load, index) in predictions.items():
        if protocol != "grouped-CV" or model.startswith("Riegel"):
            continue
        for segment in segments:
            mask = None if segment == "all" else masks[segment]
            rows.append(
                {
                    "model": model,
                    "segment": segment,
                    **paired_ape_difference(ds, without, with_load, index, mask),
                }
            )
    paired = pd.DataFrame(rows)

    banister, summary = banister_report(cleaned)

    text = (
        "# Stage 9c: training-load features and the Banister model\n\n"
        f"{len(ds)} races ({int(ds.static['is_triathlon'].sum())} triathlon runs), "
        f"{ds.meta['athlete'].nunique()} athletes. Load features: `{', '.join(LOAD_FEATURES)}` "
        "(state on the eve of the race; TRIMP load, CTL 42 d, ATL 7 d). Metrics on total race "
        "time. `grouped-CV` holds out whole athletes; `chronological` holds out each athlete's "
        "latest races.\n\n"
        "## 1. Do load features improve the models?\n\n"
        "MAPE (%) per segment, the same model with and without the load features.\n\n"
        + scores.round(2).reset_index().to_markdown(index=False)
        + "\n\n"
        "### Paired change in absolute percentage error (grouped-CV)\n\n"
        "Mean change per race in percentage points when load features are added (negative = "
        "better) with a 95 % interval from resampling athletes. `share_improved` is the fraction "
        "of races whose error went down.\n\n" + paired.round(2).to_markdown(index=False) + "\n\n"
        "## 2. How well does the Banister model describe form?\n\n"
        "Per athlete with at least 5 race days: each race is forecast from a fit on the earlier "
        "races only. Performance = Riegel-equivalent 10 km speed; errors are relative (%). "
        "Baselines: running mean of earlier races, previous race. `at_bound` = a fitted time "
        "constant sits on its limit (not identified by the data).\n\n"
        + banister.round(2).to_markdown(index=False)
        + "\n\n"
        f"Pooled over {summary['forecasts']:.0f} forecasts ({summary['athletes']:.0f} athletes): "
        f"Banister **{summary['banister_MAPE_%']:.2f} %**, running mean "
        f"**{summary['running_mean_MAPE_%']:.2f} %**, previous race "
        f"**{summary['previous_race_MAPE_%']:.2f} %**. "
        f"{summary['full_fits_at_bound']:.0f} of {summary['full_fits']:.0f} full fits have a "
        "time constant at a bound.\n\n"
    )
    write_report(args.out, text)
    print(text)


if __name__ == "__main__":
    main()
