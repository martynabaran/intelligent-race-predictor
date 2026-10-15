"""Compare GRU / TCN / Transformer encoders under both evaluation protocols (several seeds).

python experiments/run_architectures.py [--seeds 3]
"""

import argparse
from pathlib import Path

import pandas as pd

from race_predictor.evaluation.runner import compare_models, load_dataset
from race_predictor.models.sequence import SequenceModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_ARCHITECTURES.md"))
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    ds = load_dataset(args.data)
    runs = []
    for seed in range(args.seeds):
        factories = {
            name: (lambda enc=enc, seed=seed: SequenceModel(enc, seed=seed))
            for name, enc in (("GRU", "gru"), ("TCN", "tcn"), ("Transformer", "transformer"))
        }
        runs.append(compare_models(ds, factories).assign(seed=seed))
        print(f"seed {seed} done", flush=True)
    raw = pd.concat(runs)
    metrics = ["MAE_time_min", "RMSE_time_min", "MAPE_%", "MAE_pace_s_km", "fit_s"]
    mean = raw.groupby(["model", "protocol"])[metrics].mean()
    std = raw.groupby(["model", "protocol"])["MAPE_%"].std().rename("MAPE_std")
    table = pd.concat([mean, std], axis=1).round(2).reset_index()
    print(table.to_string(index=False))
    args.out.write_text(
        "# Sequence architectures: GRU vs TCN vs Transformer\n\n"
        f"{len(ds)} races, {ds.meta['athlete'].nunique()} athletes; identical hybrid layout "
        f"(encoder + static branch), mean over {args.seeds} seeds (`MAPE_std` = std of MAPE across "
        "seeds). Metrics on total race time. With so few races, differences smaller than "
        "`MAPE_std` should not be over-interpreted.\n\n" + table.to_markdown(index=False) + "\n"
    )


if __name__ == "__main__":
    main()
