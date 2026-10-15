"""Pretrain on the Kaggle marathon data, then compare fine-tuned vs. from-scratch GRU.

python scripts/download_kaggle.py --only general
python experiments/run_transfer.py --kaggle data/kaggle/general
"""

import argparse
from pathlib import Path

from race_predictor.evaluation.runner import compare_models, load_dataset
from race_predictor.features import build_marathon_data
from race_predictor.models.transfer import TransferGRU, pretrain_encoder
from race_predictor.sources.kaggle import load_marathon_activities


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--kaggle", type=Path, default=Path("data/kaggle/general"))
    ap.add_argument("--encoder", type=Path, default=Path("models/encoder.joblib"))
    ap.add_argument("--out", type=Path, default=Path("docs/RESULTS_TRANSFER.md"))
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    args = ap.parse_args()

    if args.encoder.exists():
        from race_predictor.models.transfer import PretrainedEncoder

        encoder = PretrainedEncoder.load(args.encoder)
        print(f"loaded {args.encoder}")
    else:
        pre = build_marathon_data(load_marathon_activities(args.kaggle))
        print(f"pretraining on {len(pre)} marathons from {len(set(pre.athlete))} athletes")
        encoder = pretrain_encoder(pre, epochs=args.pretrain_epochs)
        args.encoder.parent.mkdir(parents=True, exist_ok=True)
        encoder.save(args.encoder)

    ds = load_dataset(args.data)
    table = compare_models(
        ds,
        {
            "GRU-scratch": lambda: TransferGRU(),
            "GRU-finetuned": lambda: TransferGRU(pretrained=encoder),
        },
    ).round(2)
    print(table.to_string(index=False))
    args.out.write_text(
        "# Transfer learning: pretrained on Kaggle marathons vs. from scratch\n\n"
        f"{len(ds)} races. Identical architecture and shared input channels; only initialisation differs.\n\n"
        + table.to_markdown(index=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
