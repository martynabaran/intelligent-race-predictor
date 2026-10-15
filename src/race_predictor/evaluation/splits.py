"""Evaluation protocols over a :class:`RaceDataset`.

* ``chronological_split`` – thesis protocol: each athlete's most recent races are the test set
  (athletes are seen in training).
* ``grouped_splits`` – stricter: whole athletes are held out (``GroupKFold``), measuring
  generalisation to unseen runners.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold

from race_predictor.features import RaceDataset


def chronological_split(ds: RaceDataset, test_fraction: float = 0.2):
    test = np.zeros(len(ds), dtype=bool)
    for _, idx in ds.meta.groupby("athlete").groups.items():
        order = ds.meta.loc[idx].sort_values("race_date").index.to_numpy()
        test[order[len(order) - max(1, int(len(order) * test_fraction)) :]] = True
    return np.flatnonzero(~test), np.flatnonzero(test)


def grouped_splits(ds: RaceDataset, n_splits: int = 5) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    groups = ds.meta["athlete"].to_numpy()
    n_splits = min(n_splits, len(set(groups)))
    yield from GroupKFold(n_splits=n_splits).split(np.zeros(len(ds)), groups=groups)
