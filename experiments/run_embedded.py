"""Distil the GRU teacher into small students and export to TFLite (float / int8).

python experiments/run_embedded.py
"""

import argparse
from pathlib import Path

import pandas as pd

from race_predictor.evaluation.embedded import (
    StudentSpec,
    evaluate_embedded,
    measure_deployment,
    teacher_scores,
)
from race_predictor.evaluation.runner import load_dataset
from race_predictor.models.distill import CompactSequence, TabularMLP
from race_predictor.models.sequence import SequenceModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_EMBEDDED.md"))
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    teacher = lambda: SequenceModel("gru", seed=0)  # noqa: E731
    students = [
        StudentSpec("MLP(16,8) tabular", lambda: TabularMLP(), ("float", "int8")),
        # The int8 interpreter aborts on this one (-1e9 masked max) and PAD_VALUE=999 in the input
        # would swamp the int8 range anyway -> float only; CompactSequence fixes both.
        StudentSpec(
            "TCN-small (w=8)", lambda: SequenceModel("tcn", width=8, dense_units=16), ("float",)
        ),
        StudentSpec("CompactSequence (w=8)", lambda: CompactSequence(), ("float", "int8")),
    ]
    acc = pd.concat(
        [teacher_scores(ds, teacher), evaluate_embedded(ds, teacher, students, args.alpha)]
    )
    dep = measure_deployment(ds, teacher, students, args.alpha)

    acc = acc.round(2)
    wide = acc.pivot_table(
        index=["student", "training", "format"],
        columns="protocol",
        values=["MAPE_%", "MAE_time_min"],
        sort=False,
    )
    wide.columns = [f"{m} ({p})" for m, p in wide.columns]
    text = (
        "# Embedded models: distillation + TFLite quantisation\n\n"
        f"{len(ds)} races. Teacher: GRU (hybrid, as in Stage 4). Students are trained plainly or "
        f"distilled (targets = {args.alpha}*true + {1 - args.alpha}*teacher). Accuracy is on total "
        "race time with the same two protocols as before; `keras` = float model before export.\n\n"
        "## Accuracy\n\n" + wide.round(2).reset_index().to_markdown(index=False) + "\n\n"
        "## Size and latency (models fitted on all data)\n\n"
        "Latency: median single-sample inference, 1 thread, host CPU (relative indicator only).\n\n"
        + dep.round(1).to_markdown(index=False)
        + "\n"
    )
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
