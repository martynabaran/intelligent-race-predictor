"""Stage 10: which model for running races (the app's choice) - single models vs nested selection.

Triathlon runs are no longer races (39 of 41 were brick sessions, `docs/REVIEW_RACES.md`), so
there is one mode. The app uses the Riegel formula; `AutoRouter` picks the model inside every
training fold (inner athlete-grouped CV), so its scores are a nested estimate of "choose the
best model" without peeking at the test athletes.

python experiments/run_modes.py
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from race_predictor.domain import RaceMode
from race_predictor.evaluation.breakdown import breakdown
from race_predictor.evaluation.modes import AutoModeRouter
from race_predictor.evaluation.runner import load_dataset, protocol_predictions
from race_predictor.models import (
    HistGradientBoostingBaseline,
    RidgeBaseline,
    RiegelBaseline,
    RiegelCorrected,
)

SEGMENTS = ["all"]


class RecordingAutoRouter(AutoModeRouter):
    """Remembers which model every outer fold chose, for the report."""

    choices: list[dict] = []

    def fit(self, ds):
        super().fit(ds)
        RecordingAutoRouter.choices.append({m.value: n for m, n in self.chosen_.items()})
        return self


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_MODES.md"))
    args = ap.parse_args()

    ds = load_dataset(args.data)
    factories = {
        "Riegel (app)": RiegelBaseline,
        "Ridge (previous app model)": RidgeBaseline,
        "HistGB": HistGradientBoostingBaseline,
        "Riegel+correction": RiegelCorrected,
        "AutoRouter (nested selection)": RecordingAutoRouter,
    }
    rows = []
    for name, make in factories.items():
        for protocol, pred, index in protocol_predictions(ds, make):
            rows.append(breakdown(ds, pred, index).assign(model=name, protocol=protocol))
        print(f"{name} done", flush=True)
    raw = pd.concat(rows, ignore_index=True)

    table = raw[raw["segment"].isin(SEGMENTS)].pivot_table(
        index=["protocol", "model"],
        columns="segment",
        values=["MAPE_%", "MedAPE_%", "MAE_time_min"],
        sort=False,
    )
    table = table.reindex(
        columns=[(m, s) for m in ("MAPE_%", "MedAPE_%", "MAE_time_min") for s in SEGMENTS]
    )
    table.columns = [f"{m} {s}" for m, s in table.columns]
    table = table.round(2).reset_index()

    grouped = raw[raw["protocol"] == "grouped-CV"]
    per_distance = grouped[grouped["segment"].str.endswith("(stand-alone)")].pivot_table(
        index="segment", columns="model", values="MAPE_%", sort=False
    )
    per_distance.insert(0, "n", grouped.groupby("segment")["n"].first().reindex(per_distance.index))
    per_distance.index = per_distance.index.str.replace(" (stand-alone)", "")

    picks = {
        mode.value: dict(Counter(c[mode.value] for c in RecordingAutoRouter.choices))
        for mode in RaceMode
    }
    picks_md = "\n".join(f"- `{mode}`: {counts}" for mode, counts in picks.items())

    print(table.to_string(index=False), picks_md, sep="\n\n")
    args.out.write_text(
        "# Stage 10: model for running races\n\n"
        f"{len(ds)} running races of {ds.meta['athlete'].nunique()} athletes (runs after a ride "
        "are no longer races, see `REVIEW_RACES.md`). Metrics on total race time; `MedAPE` = "
        "median absolute percentage error. The app uses Riegel; `AutoRouter` re-selects the "
        "model inside every training fold (inner grouped CV over Riegel, Ridge, HistGB, "
        "Riegel+correction), so its scores are a nested estimate.\n\n"
        + table.to_markdown(index=False)
        + "\n\n## MAPE per distance (grouped-CV)\n\n"
        + per_distance.round(1).reset_index().to_markdown(index=False)
        + "\n\n"
        f"## What AutoRouter chose ({len(RecordingAutoRouter.choices)} outer fits)\n\n"
        + picks_md
        + "\n"
    )


if __name__ == "__main__":
    main()
