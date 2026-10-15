import math
from pathlib import Path

import pytest

from race_predictor.domain import ActivityType
from race_predictor.exceptions import UnsupportedExportFormatError
from race_predictor.sources import available_sources, detect_source, load_activities
from race_predictor.sources.garmin import parse_duration, speed_to_ms
from race_predictor.sources.strava import parse_strava_date

GARMIN_HEADER = "Typ aktywności,Data,Dystans,Czas,Średnie tętno,Maksymalne tętno,Średnie tempo,Najlepsze tempo,Czas ruchu,Minimalna temperatura,Maksymalna temperatura"
GARMIN_ROW = 'Bieganie,2025-09-19 21:33:31,"6,44",00:38:10,152,177,5:56,4:45,00:37:57,20.0,25.0'
STRAVA_HEADER = "Activity ID,Activity Date,Activity Name,Activity Type,Elapsed Time,Distance,Max Heart Rate,Moving Time,Average Speed,Average Heart Rate"
STRAVA_ROW = '1,"Jul 17, 2021, 3:01:35 PM",Morning Run,Run,3000,10.0,180,2900,3.4,150'


def test_registry_has_builtin_sources():
    assert {"strava", "garmin"} <= set(available_sources())


def test_parse_duration():
    assert parse_duration("01:02:03") == 3723
    assert parse_duration("5:56") == 356
    assert math.isnan(parse_duration("--"))


def test_speed_conversion():
    assert speed_to_ms("5:00", ActivityType.RUN) == pytest.approx(1000 / 300)
    assert speed_to_ms("36", ActivityType.RIDE) == pytest.approx(10)
    assert speed_to_ms("2:00", ActivityType.SWIM) == pytest.approx(100 / 120)
    assert math.isnan(speed_to_ms("--", ActivityType.RUN))


def test_strava_date():
    assert parse_strava_date("Jul 17, 2021, 3:01:35 PM").hour == 15


def test_garmin_file(tmp_path: Path):
    f = tmp_path / "g.csv"
    f.write_text(f"{GARMIN_HEADER}\n{GARMIN_ROW}\n")
    assert detect_source(f).name == "garmin"
    df = load_activities(f, "me")
    row = df.iloc[0]
    assert row["type"] == "Run" and row["distance_m"] == pytest.approx(6440)
    assert row["avg_speed_ms"] == pytest.approx(1000 / 356)
    assert row["avg_temp_c"] == pytest.approx(22.5)


def test_strava_file(tmp_path: Path):
    f = tmp_path / "s.csv"
    f.write_text(f"{STRAVA_HEADER}\n{STRAVA_ROW}\n")
    assert detect_source(f).name == "strava"
    row = load_activities(f, "me").iloc[0]
    assert row["distance_m"] == pytest.approx(10000) and row["max_hr"] == 180


def test_unsupported_format(tmp_path: Path):
    f = tmp_path / "x.csv"
    f.write_text("a,b\n1,2\n")
    with pytest.raises(UnsupportedExportFormatError):
        detect_source(f)


DATA = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_real_exports_load():
    from race_predictor.sources import load_directory

    assert load_directory(DATA / "strava_activities")["athlete"].nunique() > 1
    assert load_directory(DATA / "garmin_activities")["athlete"].nunique() == 2
