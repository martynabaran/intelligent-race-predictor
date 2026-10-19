"""Hyper-parameter optimisation with Optuna.

``tune`` optimises the pooled out-of-fold MAPE (on race time) of a model factory over
athlete-grouped folds. Because tuning on the same folds that are then reported is optimistic,
``nested_evaluate`` re-tunes inside each outer training fold and scores the held-out athletes.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna

from race_predictor.evaluation.metrics import evaluate
from race_predictor.evaluation.splits import grouped_splits
from race_predictor.features import RaceDataset
from race_predictor.models.base import RaceModel
from race_predictor.models.baselines import HistGradientBoostingBaseline

Space = Callable[[optuna.Trial], RaceModel]


def histgb_space(trial: optuna.Trial) -> RaceModel:
    return HistGradientBoostingBaseline(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_iter=trial.suggest_int("max_iter", 50, 400, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 3, 31),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
        l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    )


def gru_space(trial: optuna.Trial) -> RaceModel:
    from race_predictor.models.gru import GRUModel

    return GRUModel(
        epochs=100,
        patience=15,
        hidden_size=trial.suggest_categorical("hidden_size", [8, 16, 32]),
        dense_units=trial.suggest_categorical("dense_units", [16, 32, 64]),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
    )


def cv_mape(ds: RaceDataset, make: Callable[[], RaceModel], n_splits: int) -> float:
    """Pooled out-of-fold MAPE (%) on total race time."""
    pred, idx = [], []
    for tr, te in grouped_splits(ds, n_splits):
        pred.append(make().fit(ds.subset(tr)).predict(ds.subset(te)))
        idx.append(te)
    idx = np.concatenate(idx)
    sub = ds.subset(idx)
    km = sub.meta["race_distance_m"].to_numpy() / 1000
    return evaluate(sub.y * km, np.concatenate(pred) * km)["mape_pct"]


def tune(
    ds: RaceDataset,
    space: Space,
    n_trials: int = 30,
    n_splits: int = 3,
    seed: int = 0,
    timeout: float | None = None,
) -> optuna.Study:
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: cv_mape(ds, lambda: space(trial), n_splits),
        n_trials=n_trials,
        timeout=timeout,  # repeated suggest_* calls per fold return the same value
    )
    return study


def best_model(study: optuna.Study, space: Space) -> RaceModel:
    return space(optuna.trial.FixedTrial(study.best_params))


def nested_evaluate(
    ds: RaceDataset,
    space: Space,
    n_trials: int = 20,
    outer_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 0,
) -> dict[str, float]:
    """Unbiased estimate of tuned performance: tune on each outer training set only."""
    pred, idx = [], []
    for tr, te in grouped_splits(ds, outer_splits):
        train = ds.subset(tr)
        study = tune(train, space, n_trials, inner_splits, seed)
        pred.append(best_model(study, space).fit(train).predict(ds.subset(te)))
        idx.append(te)
    idx = np.concatenate(idx)
    sub = ds.subset(idx)
    km = sub.meta["race_distance_m"].to_numpy() / 1000
    m = evaluate(sub.y * km, np.concatenate(pred) * km)
    return {"MAE_time_min": m["mae"] / 60, "RMSE_time_min": m["rmse"] / 60, "MAPE_%": m["mape_pct"]}
