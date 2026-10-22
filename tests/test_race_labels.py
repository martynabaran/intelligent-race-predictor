import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from race_predictor.evaluation.report import CONCLUSIONS, write_report
from race_predictor.exceptions import DataValidationError
from race_predictor.processing import (
    ApplyRaceLabels,
    cleaning_pipeline,
    load_race_labels,
)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def labels(*rows):
    """Label table; rows are (athlete, 'YYYY-MM-DD', km, is_race)."""
    df = pd.DataFrame(rows, columns=["athlete", "date", "distance_km", "is_race"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def runs(frame, *rows, **defaults):
    """Activity table with ``is_race=False``; rows are dicts overriding the frame defaults."""
    return frame([{"is_race": False, **r} for r in rows])


# ----- loading --------------------------------------------------------------------------------


def test_load_race_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "athlete,date,distance_km,is_race,list\n"
        "a,2024-05-01 07:30:00,10.0,1,B\n"
        "a,2024-05-02,7.01,0,C\n"
    )
    loaded = load_race_labels(path)
    assert loaded["is_race"].tolist() == [True, False]
    assert loaded["date"].iloc[0] == pd.Timestamp("2024-05-01")  # normalised to the day
    assert "list" in loaded.columns  # extra columns are kept


def test_load_race_labels_rejects_missing_columns(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("athlete,date,is_race\na,2024-05-01,1\n")
    with pytest.raises(DataValidationError, match="missing columns"):
        load_race_labels(path)


def test_load_race_labels_rejects_contradictions(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("athlete,date,distance_km,is_race\na,2024-05-01,10.0,1\na,2024-05-01,10.0,0\n")
    with pytest.raises(DataValidationError, match="contradicting"):
        load_race_labels(path)


def test_identical_duplicate_labels_are_collapsed(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("athlete,date,distance_km,is_race\na,2024-05-01,10.0,1\na,2024-05-01,10.0,1\n")
    assert len(load_race_labels(path)) == 1


def test_step_rejects_a_table_without_the_required_columns():
    with pytest.raises(DataValidationError):
        ApplyRaceLabels(pd.DataFrame({"athlete": ["a"]}))


# ----- applying -------------------------------------------------------------------------------


def test_labels_override_the_detector_both_ways(frame, days):
    df = runs(
        frame,
        {"date": days(0), "distance_m": 10_000.0},  # detector says no, reviewer says yes
        {"date": days(1), "distance_m": 12_000.0},  # detector says yes, reviewer says no
        {"date": days(2), "distance_m": 8_000.0},  # not labelled: keeps the detector's decision
    )
    df.loc[1, "is_race"] = True
    out = ApplyRaceLabels(
        labels(("a", str(days(0).date()), 10.0, True), ("a", str(days(1).date()), 12.0, False))
    ).apply(df)
    assert out["is_race"].tolist() == [True, False, False]


def test_unlabelled_runs_keep_the_detectors_decision(frame, days):
    df = runs(frame, {"date": days(0)}, {"date": days(1)})
    df.loc[1, "is_race"] = True
    out = ApplyRaceLabels(labels(("a", str(days(5).date()), 10.0, False))).apply(df)
    assert out["is_race"].tolist() == [False, True]


def test_distance_is_matched_within_the_rounding_of_the_review(frame, days):
    day = str(days(0).date())
    df = runs(frame, {"date": days(0), "distance_m": 10_004.0})
    assert ApplyRaceLabels(labels(("a", day, 10.0, True))).apply(df)["is_race"].all()  # 4 m off
    far = runs(frame, {"date": days(0), "distance_m": 10_030.0})
    assert not ApplyRaceLabels(labels(("a", day, 10.0, True))).apply(far)["is_race"].any()


def test_two_runs_on_one_day_are_told_apart_by_distance(frame, days):
    day = str(days(0).date())
    df = runs(
        frame,
        {"date": days(0), "distance_m": 5_000.0},
        {"date": days(0) + pd.Timedelta(hours=9), "distance_m": 10_000.0},
    )
    out = ApplyRaceLabels(labels(("a", day, 10.0, True))).apply(df)
    assert out["is_race"].tolist() == [False, True]


def test_labels_apply_only_to_the_named_athlete_and_to_runs(frame, days):
    day = str(days(0).date())
    df = runs(
        frame,
        {"date": days(0)},
        {"date": days(0), "athlete": "b"},
        {"date": days(0), "type": "Ride"},
    )
    out = ApplyRaceLabels(labels(("a", day, 10.0, True))).apply(df)
    assert out["is_race"].tolist() == [True, False, False]


def test_unmatched_labels_are_reported(frame, days):
    df = runs(frame, {"date": days(0)})
    step = ApplyRaceLabels(
        labels(("a", str(days(0).date()), 10.0, True), ("a", "2020-01-01", 10.0, True))
    )
    step.apply(df)
    assert step.unmatched["date"].tolist() == [pd.Timestamp("2020-01-01")]


def test_applying_labels_does_not_modify_the_input(frame, days):
    df = runs(frame, {"date": days(0)})
    before = df.copy()
    ApplyRaceLabels(labels(("a", str(days(0).date()), 10.0, True))).apply(df)
    pd.testing.assert_frame_equal(df, before)


def test_labels_work_on_a_non_default_index(frame, days):
    df = runs(frame, {"date": days(0)}, {"date": days(1)}).set_index(pd.Index([10, 20]))
    out = ApplyRaceLabels(labels(("a", str(days(1).date()), 10.0, True))).apply(df)
    assert out["is_race"].tolist() == [False, True]


# ----- in the pipeline ------------------------------------------------------------------------


def names(pipeline):
    return [type(step).__name__ for step in pipeline.steps]


def test_labels_are_applied_after_detection_and_before_the_triathlon_drop():
    steps = names(cleaning_pipeline("train", race_labels=labels(("a", "2024-01-01", 10.0, True))))
    assert steps.index("DetectRaces") < steps.index("ApplyRaceLabels")
    assert steps.index("ApplyRaceLabels") < steps.index("DropTriathlonRaces")


def test_no_labels_means_no_label_step():
    assert "ApplyRaceLabels" not in names(cleaning_pipeline("train"))


def test_a_confirmed_race_after_a_ride_is_still_not_a_road_race(frame, days):
    """Runs after a ride stay out of the road races to predict, even if the reviewer confirmed
    them (they are triathlon run legs, which have no model of their own yet)."""
    df = frame(
        [{"date": days(i)} for i in range(5)]
        + [
            {"date": days(10), "type": "Ride", "avg_speed_ms": 8.0, "distance_m": 40_000.0},
            {"date": days(10) + pd.Timedelta(hours=2), "distance_m": 10_000.0, "avg_hr": 175.0},
        ]
    )
    run_day = str((days(10) + pd.Timedelta(hours=2)).date())
    out = cleaning_pipeline("train", race_labels=labels(("a", run_day, 10.0, True))).run(df)
    assert out["is_triathlon"].sum() == 1
    assert not out["is_race"].any()


# ----- the real data --------------------------------------------------------------------------

needs_data = pytest.mark.skipif(
    not (DATA / "race_labels.csv").exists(), reason="race labels not present"
)


@needs_data
def test_every_label_matches_a_run_in_the_data():
    """Guards against athlete ids drifting (they come from the sorted file order in data/)."""
    from race_predictor.sources import load_directory

    raw = pd.concat(
        [load_directory(DATA / "strava_activities"), load_directory(DATA / "garmin_activities")],
        ignore_index=True,
    )
    step = ApplyRaceLabels(load_race_labels(DATA / "race_labels.csv"))
    step.apply(raw.assign(is_race=False))
    assert step.unmatched.empty, step.unmatched.head()


@needs_data
def test_cleaned_training_data_follows_the_reviewers_decisions():
    from race_predictor.evaluation.runner import load_cleaned

    cleaned = load_cleaned(DATA)
    day = cleaned["date"].dt.normalize()
    km = cleaned["distance_m"] / 1000
    for row in load_race_labels(DATA / "race_labels.csv").itertuples():
        hit = cleaned[
            (cleaned["athlete"] == row.athlete)
            & (day == row.date)
            & ((km - row.distance_km).abs() <= 0.006)
            & (cleaned["type"] == "Run")
        ]
        assert len(hit) >= 1
        if not row.is_race:
            assert not hit["is_race"].any(), row
        elif not hit["is_triathlon"].any():  # confirmed, and not a run after a ride
            assert hit["is_race"].all(), row


# ----- label export and report helper ---------------------------------------------------------


def load_export_module():
    spec = importlib.util.spec_from_file_location(
        "export_race_labels", ROOT / "experiments" / "export_race_labels.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REVIEW = """# Przegląd
## A. Tylko po rowerze
| athlete   | date       |    km | after ride   | decyzja (start?)   |
|:----------|:-----------|------:|:-------------|:-------------------|
| strava_1  | 2024-05-01 |  4.91 | tak          |        nie         |
| strava_1  | 2024-05-02 | 10.02 | tak          |        tak         |
| strava_1  | 2024-05-03 |  5.00 | tak          |        ?           |
| strava_1  | 2024-05-04 |  5.00 | tak          |                    |
## C. Zwykłe
| athlete   | date       |    km | after ride   | decyzja (start?)   |
|:----------|:-----------|------:|:-------------|:-------------------|
| garmin_2  | 2025-01-02 |  7.50 |              |        tak         |
"""


def test_export_reads_decisions_and_skips_undecided_rows():
    parsed = load_export_module().parse_review(REVIEW)
    assert parsed[["athlete", "date", "distance_km", "is_race", "list"]].values.tolist() == [
        ["strava_1", "2024-05-01", 4.91, False, "A"],
        ["strava_1", "2024-05-02", 10.02, True, "A"],
        ["garmin_2", "2025-01-02", 7.5, True, "C"],
    ]


def test_report_keeps_hand_written_conclusions(tmp_path):
    path = tmp_path / "report.md"
    path.write_text(f"# Old\n\nold numbers\n\n{CONCLUSIONS}\n\nMy reading of the tables.\n")
    write_report(path, "# New\n\nnew numbers\n")
    text = path.read_text()
    assert "new numbers" in text and "old numbers" not in text
    assert text.endswith("My reading of the tables.\n")


def test_report_adds_a_conclusions_placeholder_when_there_is_none(tmp_path):
    path = tmp_path / "report.md"
    write_report(path, "# New\n")
    assert CONCLUSIONS in path.read_text()
