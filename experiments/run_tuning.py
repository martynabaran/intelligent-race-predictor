"""Optuna tuning of the gradient-boosting baseline, with an unbiased nested-CV estimate.

python experiments/run_tuning.py [--trials 20] [--gru-trials 0]
"""

import argparse
from pathlib import Path

import optuna
import pandas as pd

from race_predictor.evaluation.runner import compare_models, load_dataset
from race_predictor.evaluation.tuning import gru_space, histgb_space, nested_evaluate, tune
from race_predictor.models.baselines import HistGradientBoostingBaseline


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_TUNING.md"))
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--gru-trials", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    lines = [
        f"# Hyper-parameter tuning (Optuna)\n\n{len(ds)} races, {ds.meta['athlete'].nunique()} athletes.\n"
    ]

    default = compare_models(ds, {"HistGB (default)": HistGradientBoostingBaseline})
    grouped = default[default["protocol"] == "grouped-CV"].iloc[0]
    nested = nested_evaluate(ds, histgb_space, n_trials=args.trials)
    study = tune(ds, histgb_space, n_trials=args.trials)
    rows = [
        {
            "model": "HistGB default",
            "MAPE_%": grouped["MAPE_%"],
            "MAE_time_min": grouped["MAE_time_min"],
        },
        {
            "model": "HistGB tuned (nested CV)",
            "MAPE_%": nested["MAPE_%"],
            "MAE_time_min": nested["MAE_time_min"],
        },
    ]
    lines += [
        "Grouped (by athlete) CV. 'nested' re-tunes inside every outer training fold, so it is an "
        "unbiased estimate; the single study below is tuned on all folds and therefore optimistic.\n",
        pd.DataFrame(rows).round(2).to_markdown(index=False),
        f"\nBest params on all data: `{study.best_params}` (inner CV MAPE {study.best_value:.2f}%)\n",
    ]
    if args.gru_trials:
        gstudy = tune(ds, gru_space, n_trials=args.gru_trials)
        lines.append(
            f"GRU ({args.gru_trials} trials, **not nested - optimistic**): best inner-CV MAPE "
            f"{gstudy.best_value:.2f}% with `{gstudy.best_params}`\n"
        )
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
