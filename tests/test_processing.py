import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.processing import (
    DropAthletesWithoutHR,
    EstimateHRMax,
    HeuristicRaceDetector,
    MarkTriathlonRuns,
    Pipeline,
    RemoveAnomalies,
    RemoveImplausiblyFastRuns,
    StandardDistanceDetector,
    ValidateHeartRate,
    ValidateMaxHeartRate,
    cleaning_pipeline,
    default_pipeline,
    distance_label,
    training_windows,
)


def test_remove_anomalies(frame, days):
    df = frame(
        [
            {"date": days(0)},
            {"date": days(1), "avg_speed_ms": 9.0},
            {"date": days(2), "distance_m": 500.0},
            {"date": days(3), "type": "Ride", "avg_speed_ms": 12.0},
        ]
    )
    out = RemoveAnomalies().apply(df)
    assert len(out) == 2 and set(out["type"]) == {"Run", "Ride"}


def test_validate_hr(frame, days):
    df = frame(
        [
            {"date": days(0)},
            {"date": days(1), "avg_hr": 30.0},
            {"date": days(2), "avg_hr": float("nan")},
        ]
    )
    assert len(ValidateHeartRate().apply(df)) == 1
    assert len(ValidateHeartRate(keep_missing=True).apply(df)) == 2


def test_remove_implausibly_fast_runs_is_per_athlete(frame, days):
    df = frame(
        [
            *({"date": days(i), "avg_speed_ms": 3.0} for i in range(5)),
            {"date": days(5), "avg_speed_ms": 6.3},  # 2.1x this athlete's median: a ride
            {"date": days(6), "avg_speed_ms": 4.8},  # 1.6x: a fast but possible run
            {"date": days(7), "avg_speed_ms": 7.0, "type": "Ride"},  # rides are not touched
            *({"date": days(i), "athlete": "b", "avg_speed_ms": 4.5} for i in range(3)),
        ]
    )
    out = RemoveImplausiblyFastRuns().apply(df)
    assert len(out) == len(df) - 1 and 6.3 not in out["avg_speed_ms"].to_list()
    assert (out["athlete"] == "b").sum() == 3  # 4.5 m/s is normal for athlete b


def test_validate_max_hr_blanks_spikes_before_hrmax(frame, days):
    df = frame([{"date": days(0), "max_hr": 185.0}, {"date": days(1), "max_hr": 244.0}])
    out = EstimateHRMax().apply(ValidateMaxHeartRate().apply(df))
    assert len(out) == 2 and out["max_hr"].isna().sum() == 1
    assert (out["hr_max"] == 185.0).all()


def test_default_pipeline_switches():
    def names(p):
        return [type(s).__name__ for s in p.steps]

    assert "RemoveImplausiblyFastRuns" in names(default_pipeline())
    assert "ValidateMaxHeartRate" not in names(default_pipeline())
    assert "ValidateMaxHeartRate" in names(default_pipeline(hr_spike_filter=True))
    thesis = names(default_pipeline(fast_run_filter=False))
    assert "RemoveImplausiblyFastRuns" not in thesis and len(thesis) == 6


def test_train_and_predict_share_the_cleaning_steps():
    def names(p):
        return [type(s).__name__ for s in p.steps]

    train, predict = names(cleaning_pipeline("train")), names(cleaning_pipeline("predict"))
    shared = ["RemoveAnomalies", "RemoveImplausiblyFastRuns"]
    assert train[:2] == shared and predict[:2] == shared
    assert names(default_pipeline()) == train
    with_spikes = names(cleaning_pipeline("predict", hr_spike_filter=True))
    assert "ValidateMaxHeartRate" in with_spikes
    with pytest.raises(ValueError):
        cleaning_pipeline("serve")


def test_predict_cleaning_drops_ride_but_keeps_runs_without_hr(frame, days):
    nan = float("nan")
    df = frame(
        [
            *(
                {"date": days(i), "avg_speed_ms": 3.0, "avg_hr": nan, "max_hr": nan}
                for i in range(5)
            ),
            {"date": days(5), "avg_speed_ms": 3.1},  # a normal run with heart rate
            {"date": days(6), "avg_speed_ms": 6.3},  # a ride logged as a run
        ]
    )
    out = cleaning_pipeline("predict").run(df)
    assert len(out) == 6 and out["avg_hr"].notna().all()  # kept and imputed, not dropped
    assert 6.3 not in out["avg_speed_ms"].to_list()


def test_drop_athletes_without_hr(frame, days):
    nan = float("nan")
    df = frame([{"date": days(0)}, {"date": days(0), "athlete": "b", "avg_hr": nan, "max_hr": nan}])
    assert set(DropAthletesWithoutHR().apply(df)["athlete"]) == {"a"}


def test_estimate_hrmax(frame, days):
    df = frame([{"date": days(0), "max_hr": 170.0}, {"date": days(1), "max_hr": 185.0}])
    assert (EstimateHRMax().apply(df)["hr_max"] == 185.0).all()


def test_triathlon_marking(frame, days):
    df = frame(
        [
            {"date": days(0), "type": "Ride"},
            {"date": days(0) + pd.Timedelta(hours=2), "type": "Run"},
            {"date": days(5), "type": "Run"},
        ]
    )
    out = MarkTriathlonRuns().apply(df)
    assert out["is_triathlon"].tolist() == [0, 1, 0]


def test_distance_label():
    assert distance_label(42000) == "marathon"
    assert distance_label(5100) == "5k"
    assert distance_label(7000) == "other"


def test_standard_distance_detector(frame, days):
    df = frame([{"date": days(0), "distance_m": 10100.0}, {"date": days(1), "distance_m": 7000.0}])
    assert StandardDistanceDetector().detect(df).tolist() == [True, False]


def test_heuristic_detector_flags_hard_standard_distance(frame, days):
    easy = [
        {"date": days(i), "distance_m": 7000.0, "avg_hr": 140.0, "avg_speed_ms": 3.0}
        for i in range(0, 60, 3)
    ]
    race = {"date": days(61), "distance_m": 10000.0, "avg_hr": 175.0, "avg_speed_ms": 3.6}
    df = EstimateHRMax().apply(frame(easy + [race]))
    flags = HeuristicRaceDetector().detect(df)
    assert flags.iloc[-1] and not flags.iloc[:-1].any()


def test_heuristic_reasons_explain_the_decision(frame, days):
    df = frame(
        [
            *({"date": days(i), "distance_m": 8000.0} for i in range(5)),
            {"date": days(6), "distance_m": 10000.0, "avg_hr": 175.0},  # high HR + 10 k
            {"date": days(7), "distance_m": 7000.0},  # nothing special
        ]
    ).assign(hr_max=190.0, is_triathlon=0)
    detector = HeuristicRaceDetector()
    reasons = detector.reasons(df)
    assert reasons["is_race"].equals(detector.detect(df))
    race = reasons.iloc[5]
    assert race["high_hr"] and race["standard_distance"] and race["score"] == 3 and race["is_race"]
    assert reasons.iloc[6]["score"] == 0 and not reasons.iloc[6]["is_race"]


def test_pipeline_requires_columns():
    with pytest.raises(DataValidationError):
        Pipeline([]).run(pd.DataFrame({"athlete": []}))


def test_default_pipeline_adds_columns(frame, days):
    df = frame([{"date": days(i)} for i in range(5)])
    out = default_pipeline().run(df)
    assert {"hr_max", "is_triathlon", "is_race"} <= set(out.columns)


def test_training_windows_never_leak(frame, days):
    rows = [{"date": days(i), "is_race": False} for i in range(0, 40, 2)]
    rows.append({"date": days(41), "is_race": True})
    rows.append({"date": days(45), "is_race": False})  # after the race -> must be excluded
    rows.append({"date": days(0) - pd.Timedelta(days=400), "is_race": False})  # outside window
    windows = list(training_windows(frame(rows), days_window=365))
    assert len(windows) == 1
    race, window = windows[0]
    assert (window["date"] < race["date"]).all()
    assert (window["date"] >= race["date"] - pd.Timedelta(days=365)).all()
    assert len(window) == 20
