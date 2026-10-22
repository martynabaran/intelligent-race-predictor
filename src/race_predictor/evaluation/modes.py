"""Pick the model for each race mode by inner cross-validation (nested model selection).

``AutoModeRouter`` is itself a :class:`RaceModel`: inside ``fit`` it scores every candidate on
athlete-grouped folds of the *training* data only, keeps the best per mode and fits a
:class:`ModeRouter`. Evaluating it with the usual protocols is therefore a nested estimate - the
choice of model never sees the test athletes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from typing_extensions import override

from race_predictor.domain import RaceMode
from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import grouped_splits
from race_predictor.exceptions import DataValidationError
from race_predictor.features import RaceDataset
from race_predictor.models import (
    HistGradientBoostingBaseline,
    ModeRouter,
    RaceModel,
    RidgeBaseline,
    RiegelBaseline,
    RiegelCorrected,
    race_modes,
)

CANDIDATES: dict[str, Callable[[], RaceModel]] = {
    "Riegel": RiegelBaseline,
    "Ridge": RidgeBaseline,
    "HistGB": HistGradientBoostingBaseline,
    "Riegel+correction": RiegelCorrected,
    "Riegel+correction+ref": lambda: RiegelCorrected(use_reference_type=True),
}


class AutoModeRouter(RaceModel):
    """Per race mode, the candidate with the lowest inner-CV MAPE.

    Args:
        candidates: Name -> factory of the models to choose from (default: ``CANDIDATES``).
        n_splits: Number of athlete-grouped inner folds.

    Attributes:
        scores_: Out-of-fold MAPE (%) of every candidate per mode (set by ``fit``).
        chosen_: Name of the chosen candidate per mode (set by ``fit``).
        router_: The fitted :class:`ModeRouter` (set by ``fit``).
    """

    name = "AutoRouter"

    def __init__(
        self,
        candidates: Mapping[str, Callable[[], RaceModel]] | None = None,
        n_splits: int = 4,
    ) -> None:
        self.candidates = dict(candidates or CANDIDATES)
        self.n_splits = n_splits

    def _inner_scores(self, ds: RaceDataset) -> dict[RaceMode, dict[str, float]]:
        """MAPE (race time) of every candidate on each mode's races, out-of-fold."""
        oof = {name: np.full(len(ds), np.nan) for name in self.candidates}
        for train_idx, test_idx in grouped_splits(ds, self.n_splits):
            for name, make in self.candidates.items():
                oof[name][test_idx] = make().fit(ds.subset(train_idx)).predict(ds.subset(test_idx))
        km = ds.meta["race_distance_m"].to_numpy() / 1000
        modes = race_modes(ds)
        scores: dict[RaceMode, dict[str, float]] = {}
        for mode in RaceMode:
            in_mode = modes == mode
            if in_mode.any():
                scores[mode] = {
                    name: evaluate(ds.y[in_mode] * km[in_mode], pred[in_mode] * km[in_mode])[
                        "mape_pct"
                    ]
                    for name, pred in oof.items()
                }
        return scores

    @override
    def fit(self, ds: RaceDataset) -> AutoModeRouter:
        if len(ds) == 0:
            raise DataValidationError("cannot select models: the dataset has no races")
        self.scores_ = self._inner_scores(ds)
        road = self.scores_.get(RaceMode.ROAD) or next(iter(self.scores_.values()))
        self.chosen_ = {}
        for mode in RaceMode:
            mode_scores = self.scores_.get(mode, road)  # modes without races use the road ranking
            self.chosen_[mode] = min(mode_scores, key=lambda name: mode_scores[name])
        self.router_ = ModeRouter(
            {mode: self.candidates[name] for mode, name in self.chosen_.items()}
        ).fit(ds)
        return self

    @override
    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.router_.predict(ds)
