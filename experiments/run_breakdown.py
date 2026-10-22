"""Error analysis: effect of the data-quality steps, and errors per segment of races.

Runs the fast models on the thesis pipeline (``strict=False``) and on the cleaned one, under both
protocols, and breaks every result down into stand-alone races, triathlon runs and distances.

python experiments/run_breakdown.py
"""

import argparse
from pathlib import Path

import pandas as pd

from race_predictor.evaluation.breakdown import breakdown
from race_predictor.evaluation.runner import default_factories, load_dataset, protocol_predictions
from race_predictor.processing import default_pipeline

# This analysis is what led to dropping triathlon runs, so it keeps them.
TRI = {"include_triathlon_races": True}


def describe(ds) -> dict:
    tri = int(ds.static["is_triathlon"].sum())
    return {
        "races": len(ds),
        "athletes": ds.meta["athlete"].nunique(),
        "triathlon runs": tri,
        "stand-alone": len(ds) - tri,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_BREAKDOWN.md"))
    args = ap.parse_args()

    # "fast-run filter" keeps the thesis race set (the removed run was not a race), so it is the
    # like-for-like comparison; the HR-spike filter lowers HRmax and so changes which races exist.
    datasets = {
        "thesis": load_dataset(args.data, default_pipeline(fast_run_filter=False, **TRI)),
        "fast-run filter": load_dataset(args.data, default_pipeline(**TRI)),
        "+ HR-spike filter": load_dataset(args.data, default_pipeline(hr_spike_filter=True, **TRI)),
    }
    sizes = pd.DataFrame({k: describe(v) for k, v in datasets.items()}).T

    rows = []
    for pipeline, ds in datasets.items():
        for model, make in default_factories().items():
            for protocol, pred, index in protocol_predictions(ds, make):
                table = breakdown(ds, pred, index)
                rows.append(table.assign(model=model, protocol=protocol, pipeline=pipeline))
        print(f"{pipeline} done", flush=True)
    raw = pd.concat(rows, ignore_index=True)

    main_segments = ["all", "stand-alone", "triathlon"]
    summary = raw[raw["segment"].isin(main_segments)].pivot_table(
        index=["model", "protocol", "pipeline"],
        columns="segment",
        values=["MAPE_%", "MedAPE_%"],
        sort=False,
    )
    summary = summary.reindex(
        columns=[(m, s) for m in ("MAPE_%", "MedAPE_%") for s in main_segments]
    )
    summary.columns = [f"{m} {s}" for m, s in summary.columns]
    summary = summary.round(2).reset_index()

    cleaned = raw[(raw["pipeline"] == "fast-run filter") & (raw["protocol"] == "grouped-CV")]
    per_distance = (
        cleaned[cleaned["segment"].str.endswith("(stand-alone)")]
        .pivot_table(index="segment", columns="model", values="MAPE_%", sort=False)
        .round(2)
    )
    per_distance.insert(0, "n", cleaned.groupby("segment")["n"].first().reindex(per_distance.index))

    print(sizes.to_string(), summary.to_string(index=False), per_distance.to_string(), sep="\n\n")
    args.out.write_text(
        "# Error breakdown and data cleaning\n\n"
        "`thesis` = the thesis cleaning chain; `fast-run filter` = plus `RemoveImplausiblyFastRuns` "
        "(the new default; same races as `thesis`, so directly comparable); `+ HR-spike filter` = "
        "plus `ValidateMaxHeartRate` (opt-in), which lowers inflated HRmax and therefore detects "
        "more races - its metrics are on a different race set. Metrics on total race time; "
        "`MedAPE` = median absolute percentage error. `stand-alone` excludes triathlon runs - the "
        "app only predicts stand-alone races.\n\n"
        "## Datasets\n\n" + sizes.to_markdown() + "\n\n"
        "## MAPE / median APE by segment\n\n" + summary.to_markdown(index=False) + "\n\n"
        "## MAPE per distance (stand-alone races, fast-run filter = default pipeline, grouped-CV)\n\n"
        + per_distance.reset_index().to_markdown(index=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
