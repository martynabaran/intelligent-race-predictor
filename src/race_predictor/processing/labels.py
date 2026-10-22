"""Hand-labelled races: the reviewer's decisions override the heuristic detector.

The thesis data has no race labels, so ``HeuristicRaceDetector`` guesses. A manual review of the
detector's candidates (``docs/REVIEW_RACES.md``) found it wrong about half the time, so the
decisions are kept as data (``data/race_labels.csv``) and applied after detection.

Runs are matched by athlete, calendar day and distance. The review lists distances rounded to
0.01 km, hence the tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from race_predictor.exceptions import DataValidationError
from race_predictor.processing.pipeline import Step

LABEL_COLUMNS = ["athlete", "date", "distance_km", "is_race"]
DISTANCE_TOLERANCE_KM = 0.006  # half of the 0.01 km rounding used in the review lists


def load_race_labels(path: str | Path) -> pd.DataFrame:
    """Read a labels CSV (columns ``athlete, date, distance_km, is_race``; extra columns are kept).

    Raises:
        DataValidationError: On missing columns, or two contradicting labels for one run.
    """
    labels = pd.read_csv(path)
    missing = set(LABEL_COLUMNS) - set(labels.columns)
    if missing:
        raise DataValidationError(f"{path}: missing columns {sorted(missing)}")
    labels["date"] = pd.to_datetime(labels["date"], format="mixed").dt.normalize()
    labels["is_race"] = labels["is_race"].astype(bool)
    key = ["athlete", "date", "distance_km"]
    conflicting = labels.groupby(key)["is_race"].nunique().gt(1)
    if conflicting.any():
        raise DataValidationError(
            f"{path}: contradicting labels for {list(conflicting[conflicting].index)}"
        )
    return labels.drop_duplicates(key).reset_index(drop=True)


class ApplyRaceLabels(Step):
    """Set ``is_race`` of labelled runs to the reviewer's decision; other runs keep the detector's.

    After :meth:`apply`, ``unmatched`` lists the labels that matched no run (a sign that athlete
    ids or the data changed since the review).
    """

    def __init__(self, labels: pd.DataFrame):
        missing = set(LABEL_COLUMNS) - set(labels.columns)
        if missing:
            raise DataValidationError(f"labels are missing columns {sorted(missing)}")
        self.labels = labels
        self.unmatched = labels.iloc[0:0]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        runs = pd.DataFrame(
            {
                "row": df.index,
                "athlete": df["athlete"].to_numpy(),
                "date": df["date"].dt.normalize().to_numpy(),
                "km": (df["distance_m"] / 1000).to_numpy(),
            }
        )[df["type"].eq("Run").to_numpy()]
        pairs = runs.merge(self.labels, on=["athlete", "date"])
        pairs = pairs[(pairs["km"] - pairs["distance_km"]).abs() <= DISTANCE_TOLERANCE_KM]

        matched = self.labels.merge(
            pairs[["athlete", "date", "distance_km"]].drop_duplicates(),
            on=["athlete", "date", "distance_km"],
            how="left",
            indicator=True,
        )
        self.unmatched = self.labels[matched["_merge"].eq("left_only").to_numpy()]

        df = df.copy()
        df.loc[pairs["row"].to_numpy(), "is_race"] = pairs["is_race"].to_numpy()
        return df
