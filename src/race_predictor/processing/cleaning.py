"""One definition of data cleaning, shared by training and prediction.

Before, training used ``default_pipeline()`` and the app used its own chain in
``predictor.prepare_activities``, so a data fix (e.g. dropping a ride logged as a run) reached only
one of them and the model saw differently cleaned data in the app than in training. Both now come
from :func:`cleaning_pipeline`: the shared steps are identical, and only what has to differ does.

* ``train``: drop athletes without heart rate, then add ``hr_max``, ``is_triathlon``, ``is_race``
  (race detection needs heart rate), apply hand-labelled races if given, and keep runs after a
  ride out of the road races to predict (``DropTriathlonRaces``) unless
  ``include_triathlon_races`` - only the error analysis needs them.
* ``predict``: keep rows without heart rate (users may not wear a strap), detect races with the
  same detector (needed to know whether Riegel's reference run was a race) and impute gaps after
  detection, so imputed heart rates never count as race efforts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from race_predictor.processing.labels import ApplyRaceLabels
from race_predictor.processing.pipeline import (
    DropAthletesWithoutHR,
    DropTriathlonRaces,
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

if TYPE_CHECKING:
    from race_predictor.processing.races import RaceDetector

Purpose = Literal["train", "predict"]


def shared_steps(*, fast_run_filter: bool = True, hr_spike_filter: bool = True) -> list[Step]:
    """Return the steps applied to every dataset, for training and prediction alike."""
    steps: list[Step] = [RemoveAnomalies()]
    if fast_run_filter:
        steps.append(RemoveImplausiblyFastRuns())
    if hr_spike_filter:
        steps.append(ValidateMaxHeartRate())
    return steps


def cleaning_pipeline(
    purpose: Purpose = "train",
    detector: RaceDetector | None = None,
    *,
    fast_run_filter: bool = True,
    hr_spike_filter: bool = True,
    include_triathlon_races: bool = False,
    race_labels: pd.DataFrame | None = None,
) -> Pipeline:
    """Build the cleaning chain for ``purpose``.

    ``fast_run_filter`` (``RemoveImplausiblyFastRuns``) only drops non-runs, so the detected races
    are unchanged. ``hr_spike_filter`` (``ValidateMaxHeartRate``) lowers inflated HRmax, which
    makes the HR-based race detector find more races; the manual review confirmed all 33 of those
    races (``docs/REVIEW_RACES.md``), so it is on by default. ``race_labels`` (hand-labelled
    races, :func:`~race_predictor.processing.labels.load_race_labels`) override the detector's
    decision for the runs they name.
    """
    if purpose not in ("train", "predict"):
        raise ValueError("purpose must be 'train' or 'predict'")
    from race_predictor.processing.races import DetectRaces

    labelling = [
        EstimateHRMax(),
        MarkTriathlonRuns(),
        DetectRaces(detector),
        *([] if race_labels is None else [ApplyRaceLabels(race_labels)]),
        *([] if include_triathlon_races else [DropTriathlonRaces()]),
    ]
    shared = shared_steps(fast_run_filter=fast_run_filter, hr_spike_filter=hr_spike_filter)
    if purpose == "predict":
        return Pipeline(
            [*shared, ValidateHeartRate(keep_missing=True), *labelling, ImputeMissing()]
        )
    return Pipeline([*shared, ValidateHeartRate(), DropAthletesWithoutHR(), *labelling])


def default_pipeline(
    detector: RaceDetector | None = None,
    *,
    fast_run_filter: bool = True,
    hr_spike_filter: bool = True,
    include_triathlon_races: bool = False,
    race_labels: pd.DataFrame | None = None,
) -> Pipeline:
    """Build the training chain, same as ``cleaning_pipeline("train", ...)``.

    ``fast_run_filter=False, hr_spike_filter=False, include_triathlon_races=True`` reproduces the
    thesis chain exactly.
    """
    return cleaning_pipeline(
        "train",
        detector,
        fast_run_filter=fast_run_filter,
        hr_spike_filter=hr_spike_filter,
        include_triathlon_races=include_triathlon_races,
        race_labels=race_labels,
    )
