"""Loader for the Kaggle *long-distance-running-dataset* (``run_ww_2019_d.csv`` / ``..._2020_d``).

Columns: athlete, datetime, distance [km], duration [min], gender, age_group, country, major
(comma-separated list of majors the athlete ran). Logic ported from
``notebooks/marathon_model.ipynb``.
The files are large (10.7 M rows), so they are streamed in chunks and reduced to athletes who ran
a 2019/2020 major.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from race_predictor.exceptions import DataValidationError

KAGGLE_FILES = ("run_ww_2019_d.csv", "run_ww_2020_d.csv")
MARATHON_DATES = {
    "BERLIN 2019": "2019-09-29",
    "BOSTON 2019": "2019-04-15",
    "CHICAGO 2019": "2019-10-13",
    "LONDON 2019": "2019-04-28",
    "NEW YORK 2019": "2019-11-03",
    "TOKYO 2019": "2019-03-03",
    "TOKYO 2020": "2020-03-01",
    "BOSTON 2020": "2020-09-14",
}
_COLUMNS = ["athlete", "datetime", "distance", "duration", "major"]


def iter_kaggle_chunks(directory: str | Path, chunksize: int = 1_000_000) -> Iterator[pd.DataFrame]:
    """Yield chunks restricted to athletes' rows that mention a 2019/2020 major.

    Athlete ids of the 2020 file are offset so they do not collide with 2019 ids (as in the
    thesis notebook).
    """
    directory = Path(directory)
    offset = 0
    for name in KAGGLE_FILES:
        path = directory / name
        if not path.exists():
            raise DataValidationError(f"{path} not found - run scripts/download_kaggle.py")
        for chunk in pd.read_csv(path, usecols=lambda c: c in _COLUMNS, chunksize=chunksize):
            chunk = chunk[chunk["major"].str.contains("2019|2020", na=False)].copy()
            chunk["athlete"] = chunk["athlete"] + offset
            yield chunk
        offset = int(pd.read_csv(path, usecols=["athlete"])["athlete"].max()) + 1


def load_marathon_activities(directory: str | Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    frames = list(iter_kaggle_chunks(directory, chunksize))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_COLUMNS)
