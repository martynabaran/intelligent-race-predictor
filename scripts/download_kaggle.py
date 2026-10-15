"""Download the project's Kaggle datasets into ``data/kaggle/`` (needs Kaggle credentials).

pip install -e ".[kaggle]"   # then put your API token in ~/.kaggle/kaggle.json
python scripts/download_kaggle.py [--only general]
"""

import argparse
from pathlib import Path

DATASETS = {
    "general": "mexwell/long-distance-running-dataset",
    "garmin": "zuskakluska/activities-garmin",
    "strava": "zuskakluska/strava-activities",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=DATASETS, action="append", help="dataset(s) to fetch")
    ap.add_argument("--out", type=Path, default=Path("data/kaggle"))
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    for key in args.only or DATASETS:
        target = args.out / key
        target.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATASETS[key]} -> {target}")
        api.dataset_download_files(DATASETS[key], path=str(target), unzip=True)


if __name__ == "__main__":
    main()
