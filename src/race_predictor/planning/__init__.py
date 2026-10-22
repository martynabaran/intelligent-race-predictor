"""Rule-based planning from a predicted time. Imports no models: it takes plain numbers."""

from race_predictor.planning.feasibility import GoalAssessment, assess_goal
from race_predictor.planning.zones import (
    PaceRange,
    equivalent_time,
    training_zones,
    vdot,
    zones_from_race,
)

__all__ = [
    "GoalAssessment",
    "PaceRange",
    "assess_goal",
    "equivalent_time",
    "training_zones",
    "vdot",
    "zones_from_race",
]
