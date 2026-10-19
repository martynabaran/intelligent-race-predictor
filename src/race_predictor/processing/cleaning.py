"""One definition of data cleaning, shared by training and prediction.

Before, training used ``default_pipeline()`` and the app used its own chain in
``predictor.prepare_activities``, so a data fix (e.g. dropping a ride logged as a run) reached only
one of them and the model saw differently cleaned data in the app than in training. Both now come
from :func:`cleaning_pipeline`: the shared steps are identical, and only what has to differ does.

* ``train``: drop athletes without heart rate, then add ``hr_max``, ``is_triathlon``, ``is_race``
  (race detection needs heart rate).
* ``predict``: keep rows without heart rate (users may not wear a strap) and impute gaps.
"""

from __future__ import annotations

from typing import Literal

from race_predictor.processing.pipeline import (
    DropAthletesWithoutHR,
    EstimateHRMax,
    ImputeMissing,
    MarkTriathlonRuns,
    Pipeline,
    RemoveAnomalies,
    RemoveImplausiblyFastRuns,
    Step,
    ValidateHeartRate,
    ValidateMaxHeartRate,
)

Purpose = Literal["train", "predict"]


def shared_steps(fast_run_filter: bool = True, hr_spike_filter: bool = False) -> list[Step]:
    """Steps applied to every dataset, for training and prediction alike."""
    steps: list[Step] = [RemoveAnomalies()]
    if fast_run_filter:
        steps.append(RemoveImplausiblyFastRuns())
    if hr_spike_filter:
        steps.append(ValidateMaxHeartRate())
    return steps


def cleaning_pipeline(
    purpose: Purpose = "train",
    detector=None,
    fast_run_filter: bool = True,
    hr_spike_filter: bool = False,
) -> Pipeline:
    """Cleaning chain for ``purpose``.

    ``fast_run_filter`` (``RemoveImplausiblyFastRuns``) only drops non-runs, so the detected races
    are unchanged. ``hr_spike_filter`` (``ValidateMaxHeartRate``) lowers inflated HRmax, which makes
    the HR-based race detector find more races; it is off until those races are validated.
    """
    shared = shared_steps(fast_run_filter, hr_spike_filter)
    if purpose == "predict":
        return Pipeline([*shared, ValidateHeartRate(keep_missing=True), ImputeMissing()])
    if purpose != "train":
        raise ValueError("purpose must be 'train' or 'predict'")
    from race_predictor.processing.races import DetectRaces

    return Pipeline(
        [
            *shared,
            ValidateHeartRate(),
            DropAthletesWithoutHR(),
            EstimateHRMax(),
            MarkTriathlonRuns(),
            DetectRaces(detector),
        ]
    )


def default_pipeline(
    detector=None, fast_run_filter: bool = True, hr_spike_filter: bool = False
) -> Pipeline:
    """Training chain (thesis chain + data-quality steps). Both filters ``False`` reproduces the
    thesis chain exactly. Same as ``cleaning_pipeline("train", ...)``."""
    return cleaning_pipeline("train", detector, fast_run_filter, hr_spike_filter)
