"""Data loading and evaluation entry points for the user interfaces."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from race_predictor.sources import available_sources, load_activities

__all__ = ["available_sources", "compare_baselines", "load_activities"]


def compare_baselines(data_dir: Path, *, with_gru: bool = False) -> tuple[pd.DataFrame, int, int]:
    """Compare baseline models under both protocols.

    Returns:
        ``(table, n_races, n_athletes)``.
    """
    from race_predictor.evaluation.runner import compare_models, default_factories, load_dataset

    ds = load_dataset(data_dir)
    table = compare_models(ds, default_factories(with_gru=with_gru))
    return table, len(ds), ds.meta["athlete"].nunique()
