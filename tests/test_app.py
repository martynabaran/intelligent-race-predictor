from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from race_predictor.cli import app
from race_predictor.domain import RaceMode
from race_predictor.exceptions import DataValidationError
from race_predictor.features import build_query_dataset, build_race_dataset
from race_predictor.predictor import RacePredictor, format_time, prepare_activities

DATA = Path(__file__).parent.parent / "data"
APP = DATA.parent / "src" / "race_predictor" / "app" / "streamlit_app.py"
needs_data = pytest.mark.skipif(not DATA.exists(), reason="sample data not present")


def make_athlete(frame, speed=3.2, n=40, athlete="u"):
    rows = [
        {
            "athlete": athlete,
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=5 * i),
            "distance_m": 8000.0 + 100 * (i % 5),
            "avg_speed_ms": speed,
        }
        for i in range(n)
    ]
    return frame(rows)


def synthetic_training(frame) -> pd.DataFrame:
    rows = []
    for a, speed in zip("abcdef", (2.8, 3.1, 3.4, 3.7, 4.0, 4.3), strict=True):
        start = pd.Timestamp("2023-01-01")
        for i in range(30):
            rows.append(
                {
                    "athlete": a,
                    "date": start + pd.Timedelta(days=7 * i),
                    "distance_m": 8000.0,
                    "avg_speed_ms": speed,
                    "is_race": False,
                    "is_triathlon": 0,
                }
            )
        for j in range(4):
            rows.append(
                {
                    "athlete": a,
                    "date": start + pd.Timedelta(days=215 + 28 * j),
                    "distance_m": 10000.0,
                    "avg_speed_ms": speed * 1.1,
                    "moving_s": 10000 / (speed * 1.1),
                    "is_race": True,
                    "is_triathlon": 0,
                }
            )
    return frame(rows)


def test_format_time():
    assert format_time(3723) == "1:02:03" and format_time(float("nan")) == "-"


def test_query_dataset_uses_only_past_runs(frame):
    df = make_athlete(frame)
    race_date = df["date"].iloc[20]
    ds = build_query_dataset(df, [5000.0, 10000.0], race_date)
    assert len(ds) == 2
    assert all((seq[:, 0] > 0).all() for seq in ds.seqs)  # days_to_race strictly positive
    assert ds.static["race_distance_m"].tolist() == [5000.0, 10000.0]


def test_query_dataset_none_when_history_too_short(frame):
    assert build_query_dataset(make_athlete(frame, n=2), [5000.0]) is None


def test_refactor_keeps_race_dataset_unchanged(frame):
    ds = build_race_dataset(synthetic_training(frame))
    assert len(ds) == 24 and ds.static["race_distance_m"].eq(10000.0).all()


def test_predictor_end_to_end(frame):
    train = build_race_dataset(synthetic_training(frame))
    predictor = RacePredictor(n_splits=3).fit(train)
    slow, fast = make_athlete(frame, 2.8), make_athlete(frame, 4.2)
    t_slow, t_fast = predictor.predict(slow), predictor.predict(fast)

    assert list(t_slow["distance"]) == ["5 km", "10 km", "Half marathon", "Marathon"]
    assert (t_slow["time_s"].diff().dropna() > 0).all()  # longer distance -> longer time
    assert (t_slow["time_s"] > t_fast["time_s"]).all()  # slower athlete -> slower prediction
    # Interval = prediction x empirical 10-90 % error ratios of road races. (In this synthetic
    # data races are always 10 % faster than training, so Riegel is biased and the interval need
    # not contain the point prediction; on the real data it does: ~0.93-1.16.)
    low, high = predictor.ratios_[RaceMode.ROAD]
    assert low < high
    assert (t_slow["low_s"] / t_slow["time_s"]).to_numpy() == pytest.approx(low)
    assert (t_slow["high_s"] / t_slow["time_s"]).to_numpy() == pytest.approx(high)
    assert t_slow["riegel_s"].notna().all()

    query = build_query_dataset(prepare_activities(slow), [10000.0])
    contrib = predictor.explain(slow, 10000.0)  # road race -> Riegel + learned correction
    assert predictor.model_name() == "Riegel+correction+ref" and "best_run_pace" in contrib.index
    assert contrib.sum() == pytest.approx(predictor._pace(query)[0], rel=1e-6)
    ref = predictor.reference_run(slow)
    assert ref["pace_s_km"] == pytest.approx(contrib["best_run_pace"])
    curve = predictor.form_curve(slow, step_days=30, warmup_days=60)
    assert len(curve) > 3 and np.isfinite(curve["time_s"]).all()


def test_predictor_rejects_short_history(frame):
    predictor = RacePredictor(n_splits=3).fit(build_race_dataset(synthetic_training(frame)))
    with pytest.raises(DataValidationError):
        predictor.predict(make_athlete(frame, n=2))


def test_prediction_knows_whether_reference_run_was_a_race(frame):
    predictor = RacePredictor(n_splits=3).fit(build_race_dataset(synthetic_training(frame)))
    easy = make_athlete(frame)  # steady training, no race efforts
    raced = pd.concat(
        [
            easy,
            frame(
                [
                    {
                        "athlete": "u",
                        "date": pd.Timestamp("2024-06-01"),
                        "distance_m": 10000.0,
                        "avg_speed_ms": 3.6,
                        "moving_s": 10000 / 3.6,
                        "avg_hr": 175.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    assert not prepare_activities(easy)["is_race"].any()
    assert prepare_activities(raced)["is_race"].sum() == 1  # detected in the user's own export
    assert predictor.reference_run(easy)["is_race"] is False
    assert predictor.reference_run(raced)["is_race"] is True


def test_prepare_activities_keeps_rows_without_heart_rate(frame):
    df = make_athlete(frame, n=5).assign(avg_hr=np.nan, max_hr=np.nan)
    assert len(prepare_activities(df)) == 5


runner = CliRunner()


def test_cli_sources():
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0 and "strava" in result.output and "garmin" in result.output


def test_cli_rejects_unsupported_file(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("a,b\n1,2\n")
    result = runner.invoke(app, ["predict", str(bad), "--data", str(tmp_path)])
    assert result.exit_code == 1 and "not a supported" in result.output


@needs_data
def test_cli_predict_sample():
    result = runner.invoke(
        app, ["predict", str(DATA / "garmin_activities" / "Activities Jola.csv")]
    )
    assert result.exit_code == 0, result.output
    assert "Marathon" in result.output and "Riegel" in result.output


@needs_data
def test_streamlit_app_renders_sample(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("RACE_PREDICTOR_DATA", str(DATA))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert any("Upload an export" in i.value for i in at.info)

    sample = next(s for s in at.sidebar.selectbox if "sample" in s.label)
    sample.select(sample.options[1]).run()
    assert not at.exception, at.exception
    assert len(at.metric) == 4 and at.metric[3].label == "Marathon"
    assert any("fastest run" in m.value for m in at.markdown)  # road race -> Riegel explanation

    assert any("learned correction" in c.value for c in at.sidebar.caption)
    assert any("Learned correction" in m.value for m in at.markdown)
    assert not at.sidebar.radio  # no race-type choice: running races only


def test_prediction_contract(frame):
    from race_predictor import predictor as alias
    from race_predictor.services import PREDICTION_COLUMNS, Prediction
    from race_predictor.services import RacePredictor as Service

    assert alias.RacePredictor is Service  # old import path still works
    p = Service(n_splits=3).fit(build_race_dataset(synthetic_training(frame)))
    athlete = make_athlete(frame)
    items = p.predictions(athlete)
    assert all(isinstance(i, Prediction) for i in items) and len(items) == 4
    assert items[0].mode is RaceMode.ROAD and items[0].model == "Riegel+correction+ref"
    table = p.predict(athlete)
    assert list(table.columns) == PREDICTION_COLUMNS
    assert table["mode"].eq(RaceMode.ROAD).all()
    assert table["model"].eq("Riegel+correction+ref").all()
