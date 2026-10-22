"""Common interface: every model predicts race pace [s/km] for a :class:`RaceDataset`."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from race_predictor.features import RaceDataset


class RaceModel(ABC):
    """A model that is fitted on a :class:`RaceDataset` and predicts its race paces.

    Attributes:
        name: Short label used in result tables.
    """

    name: str

    @abstractmethod
    def fit(self, ds: RaceDataset) -> RaceModel:
        """Fit on the races of ``ds`` and return ``self``."""

    @abstractmethod
    def predict(self, ds: RaceDataset) -> np.ndarray:
        """Predict the race pace [s/km] of every race in ``ds``."""

    def contributions(self, ds: RaceDataset) -> pd.DataFrame:
        """Split each predicted pace [s/km] exactly into named parts, one column per part.

        Optional capability: only models that can attribute their prediction (Riegel, ridge and
        their combinations) implement it.

        Args:
            ds: Races to explain.

        Returns:
            One row per race; the columns of a row sum to the predicted pace.

        Raises:
            NotImplementedError: If the model offers no exact attribution.
        """
        raise NotImplementedError(f"{type(self).__name__} has no exact contributions")
