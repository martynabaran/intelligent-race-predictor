from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent
PAGE = ROOT / "src" / "race_predictor" / "app" / "pages" / "1_Training_load.py"


def sample_activities() -> pd.DataFrame:
    n = 120
    return pd.DataFrame(
        {
            "athlete": "you",
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "type": "Run",
            "distance_m": 8_000.0,
            "elapsed_s": 2_700.0,
            "moving_s": 2_700.0,
            "avg_speed_ms": 3.0,
            "max_speed_ms": 4.0,
            "avg_hr": 150.0,
            "max_hr": 185.0,
            "elevation_gain_m": 20.0,
            "elevation_loss_m": 20.0,
            "avg_temp_c": 12.0,
        }
    )


def test_page_asks_for_data_first():
    at = AppTest.from_file(str(PAGE), default_timeout=60).run()
    assert not at.exception
    assert any("main page first" in i.value for i in at.info)
    assert len(at.metric) == 0


def test_page_shows_fitness_fatigue_and_form():
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.session_state["activities"] = sample_activities()
    at.run()
    assert not at.exception, at.exception
    assert [m.label for m in at.metric] == ["Fitness (CTL)", "Fatigue (ATL)", "Form (TSB)"]


def test_page_reports_unusable_data():
    bad = sample_activities().assign(avg_speed_ms=12.0)  # impossible speeds: cleaned away
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.session_state["activities"] = bad
    at.run()
    assert not at.exception
    assert any("no usable activities" in e.value for e in at.error)
