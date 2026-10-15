"""Common interface: every model predicts race pace [s/km] for a :class:`RaceDataset`."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from race_predictor.features import RaceDataset


class RaceModel(ABC):
    name: str

    @abstractmethod
    def fit(self, ds: RaceDataset) -> RaceModel: ...

    @abstractmethod
    def predict(self, ds: RaceDataset) -> np.ndarray: ...
