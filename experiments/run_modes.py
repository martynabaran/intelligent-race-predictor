"""Stage 10: one model per race mode vs. single models.

`Router` is the fixed choice used by the app (Riegel for road races, ridge for triathlon runs).
It was picked after looking at `docs/RESULTS_BREAKDOWN.md`, so `AutoRouter` repeats the choice
inside every training fold (inner athlete-grouped CV) - its scores are a nested estimate.

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
    ModeRouter,
    RidgeBaseline,
    RiegelBaseline,
    RiegelCorrected,
)

SEGMENTS = ["all", "stand-alone", "triathlon"]


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
        "Riegel": RiegelBaseline,
        "Ridge (previous app model)": RidgeBaseline,
        "HistGB": HistGradientBoostingBaseline,
        "Riegel+correction": RiegelCorrected,
        "Router (road=Riegel, triathlon=Ridge)": ModeRouter,
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

    picks = {
        mode.value: dict(Counter(c[mode.value] for c in RecordingAutoRouter.choices))
        for mode in RaceMode
    }
    picks_md = "\n".join(f"- `{mode}`: {counts}" for mode, counts in picks.items())

    print(table.to_string(index=False), picks_md, sep="\n\n")
    args.out.write_text(
        "# Stage 10: model per race mode\n\n"
        f"{len(ds)} races ({int(ds.static['is_triathlon'].sum())} triathlon runs), "
        f"{ds.meta['athlete'].nunique()} athletes. Metrics on total race time; `MedAPE` = median "
        "absolute percentage error. `Router` is the fixed choice used by the app; `AutoRouter` "
        "re-selects the model per mode inside every training fold (inner grouped CV over Riegel, "
        "Ridge, HistGB, Riegel+correction), so its scores are a nested, unbiased estimate of "
        "the routing idea.\n\n" + table.to_markdown(index=False) + "\n\n"
        f"## What AutoRouter chose ({len(RecordingAutoRouter.choices)} outer fits)\n\n"
        + picks_md
        + "\n"
    )


if __name__ == "__main__":
    main()
