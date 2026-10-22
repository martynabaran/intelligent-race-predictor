"""Pick the model for each race mode by inner cross-validation (nested model selection).

``AutoModeRouter`` is itself a :class:`RaceModel`: inside ``fit`` it scores every candidate on
athlete-grouped folds of the *training* data only, keeps the best per mode and fits a
:class:`ModeRouter`. Evaluating it with the usual protocols is therefore a nested estimate - the
choice of model never sees the test athletes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from race_predictor.domain import RaceMode
from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import grouped_splits
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
    name = "AutoRouter"

    def __init__(
        self,
        candidates: Mapping[str, Callable[[], RaceModel]] | None = None,
        n_splits: int = 4,
    ):
        self.candidates = dict(candidates or CANDIDATES)
        self.n_splits = n_splits

    def _inner_scores(self, ds: RaceDataset) -> dict[RaceMode, dict[str, float]]:
        """MAPE (race time) of every candidate on each mode's races, out-of-fold."""
        oof = {name: np.full(len(ds), np.nan) for name in self.candidates}
        for tr, te in grouped_splits(ds, self.n_splits):
            for name, make in self.candidates.items():
                oof[name][te] = make().fit(ds.subset(tr)).predict(ds.subset(te))
        km = ds.meta["race_distance_m"].to_numpy() / 1000
        modes = race_modes(ds)
        scores: dict[RaceMode, dict[str, float]] = {}
        for mode in RaceMode:
            sel = modes == mode
            if sel.any():
                scores[mode] = {
                    name: evaluate(ds.y[sel] * km[sel], pred[sel] * km[sel])["mape_pct"]
                    for name, pred in oof.items()
                }
        return scores

    def fit(self, ds: RaceDataset) -> AutoModeRouter:
        self.scores_ = self._inner_scores(ds)
        road = self.scores_.get(RaceMode.ROAD) or next(iter(self.scores_.values()))
        self.chosen_ = {
            mode: min(self.scores_.get(mode, road), key=self.scores_.get(mode, road).get)
            for mode in RaceMode
        }
        self.router_ = ModeRouter(
            {mode: self.candidates[name] for mode, name in self.chosen_.items()}
        ).fit(ds)
        return self

    def predict(self, ds: RaceDataset) -> np.ndarray:
        return self.router_.predict(ds)
