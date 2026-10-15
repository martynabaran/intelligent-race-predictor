"""Knowledge distillation + TFLite: accuracy vs. size vs. latency.

For every split the teacher is fitted once and shared by the distilled student and its TFLite
exports, so all rows of a split see the same teacher.

python experiments/run_distillation.py [--teacher tcn] [--seeds 1]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from race_predictor.evaluation.runner import _score, load_dataset
from race_predictor.evaluation.splits import chronological_split, grouped_splits
from race_predictor.models.compact import CompactModel, DistilledModel
from race_predictor.models.sequence import SequenceModel
from race_predictor.models.tflite import QUANTIZATIONS, TFLiteRaceModel, latency_ms


def fit_split(train, teacher_encoder, seed):
    """All models for one training split, keyed by row name."""
    teacher = SequenceModel(teacher_encoder, seed=seed).fit(train)
    distilled = DistilledModel(seed=seed).fit(train, teacher=teacher)
    models = {
        f"Teacher ({teacher.name})": teacher,
        "Compact (no distillation)": CompactModel(seed=seed).fit(train),
        "Distilled": distilled,
        f"Teacher ({teacher.name}) TFLite-float32": TFLiteRaceModel.from_fitted(
            teacher, "float32", None
        ),
    }
    for q in QUANTIZATIONS:
        models[f"Distilled TFLite-{q}"] = TFLiteRaceModel.from_fitted(distilled, q, train)
    return models


def footprint(models, sample) -> pd.DataFrame:
    """Parameters, size and single-race latency of the models fitted on one split."""
    rows = []
    for name, m in models.items():
        if isinstance(m, TFLiteRaceModel):
            params, size_kb = m.fitted_.model_.count_params(), m.size_kb
        else:
            params = m.model_.count_params()
            size_kb = params * 4 / 1024  # float32 weights
        rows.append(
            {
                "model": name,
                "params": params,
                "size_kb": round(size_kb, 1),
                "latency_ms": round(latency_ms(lambda m=m: m.predict(sample)), 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_DISTILLATION.md"))
    ap.add_argument("--teacher", default="tcn", choices=["gru", "tcn", "transformer"])
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    rows, sizes = [], None
    for seed in range(args.seeds):
        protocols = {
            "chronological": [chronological_split(ds)],
            "grouped-CV": list(grouped_splits(ds)),
        }
        for protocol, splits in protocols.items():
            preds, order = defaultdict(list), []
            for tr, te in splits:
                models = fit_split(ds.subset(tr), args.teacher, seed)
                for name, m in models.items():
                    preds[name].append(m.predict(ds.subset(te)))
                order.append(te)
                if sizes is None:
                    sizes = footprint(models, ds.subset(te[:1]))
            order = np.concatenate(order)
            for name, p in preds.items():
                rows.append(
                    {"model": name, "protocol": protocol, "seed": seed,
                     **_score(ds, np.concatenate(p), order)}
                )
            print(f"seed {seed} {protocol} done", flush=True)

    raw = pd.DataFrame(rows)
    metrics = ["MAE_time_min", "RMSE_time_min", "MAPE_%", "MAE_pace_s_km"]
    grouped = raw.groupby(["model", "protocol"], sort=False)
    std = grouped["MAPE_%"].std().rename("MAPE_std")
    acc = pd.concat([grouped[metrics].mean(), std], axis=1).round(2).reset_index()
    print(acc.to_string(index=False))
    print(sizes.to_string(index=False))
    args.out.write_text(
        "# Knowledge distillation and TFLite\n\n"
        f"{len(ds)} races, {ds.meta['athlete'].nunique()} athletes; teacher: `{args.teacher}`, "
        f"mean over {args.seeds} seed(s). The teacher is fitted once per split and shared by "
        "the distilled student and its TFLite exports. Metrics on total race time; `MAPE_std` = "
        "std of MAPE across seeds - differences smaller than it should not be over-interpreted."
        "\n\n"
        "## Accuracy\n\n" + acc.to_markdown(index=False) + "\n\n"
        "## Footprint\n\nSingle-race CPU latency (median of 100 runs, including preprocessing); "
        "Keras size is float32 weights, TFLite size is the flatbuffer.\n\n"
        + sizes.to_markdown(index=False) + "\n"
    )


if __name__ == "__main__":
    main()
