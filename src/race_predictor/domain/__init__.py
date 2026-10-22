"""Domain model: activities, athletes, races and goals."""

from race_predictor.domain.distances import MARATHON_M
from race_predictor.domain.goal import Goal, Verdict
from race_predictor.domain.models import Activity, ActivityType, Athlete, Race, RaceMode

__all__ = [
    "MARATHON_M",
    "Activity",
    "ActivityType",
    "Athlete",
    "Goal",
    "Race",
    "RaceMode",
    "Verdict",
]
