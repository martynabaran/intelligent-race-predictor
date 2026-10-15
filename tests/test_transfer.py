import numpy as np
import pandas as pd
import pytest

from race_predictor.exceptions import DataValidationError
from race_predictor.features import SHARED_SEQ_FEATURES, build_marathon_data, build_race_dataset
from race_predictor.models import RidgeBaseline
from race_predictor.models.tuning import histgb_space, nested_evaluate, tune
from race_predictor.sources.kaggle import iter_kaggle_chunks, load_marathon_activities


def kaggle_frame(n_athletes=6, seed=0) -> pd.DataFrame:
    """Synthetic data in the Kaggle schema: 40 weekly runs, then a London 2019 marathon."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_athletes):
        speed_min_km = rng.uniform(4.5, 6.5)
        for w in range(40):
            d = float(rng.uniform(6, 16))
            rows.append(
                dict(
                    athlete=a,
                    datetime=pd.Timestamp("2019-04-28") - pd.Timedelta(days=7 * (w + 1)),
                    distance=d,
                    duration=d * speed_min_km * 1.03,
                    major="LONDON 2019",
                )
            )
        rows.append(
            dict(
                athlete=a,
                datetime=pd.Timestamp("2019-04-28"),
                distance=42.5,
                duration=42.195 * speed_min_km * 1.1,
                major="LONDON 2019",
            )
        )
    return pd.DataFrame(rows)


def test_marathon_data_excludes_race_and_has_shared_channels():
    data = build_marathon_data(kaggle_frame())
    assert len(data) == 6
    assert all(s.shape[1] == len(SHARED_SEQ_FEATURES) and (s[:, 0] > 0).all() for s in data.seqs)
    assert data.y.min() > 200  # s/km, plausible marathon pace


def test_marathon_data_drops_implausible_times():
    df = kaggle_frame(2)
    df.loc[df["distance"] > 42, "duration"] = 600.0
    assert len(build_marathon_data(df)) == 0


def test_kaggle_chunks_offset_and_filter(tmp_path):
    for year, base in ((2019, 0), (2020, 0)):
        df = kaggle_frame(2)
        df["athlete"] += base
        df.loc[0, "major"] = None
        df.to_csv(tmp_path / f"run_ww_{year}_d.csv", index=False)
    out = load_marathon_activities(tmp_path, chunksize=50)
    assert out["athlete"].nunique() == 4  # 2020 ids offset -> no collision
    assert out["major"].notna().all()


def test_missing_kaggle_files(tmp_path):
    with pytest.raises(DataValidationError):
        list(iter_kaggle_chunks(tmp_path))


def test_nested_tuning_runs(frame):
    rows = []
    for a, speed in zip("abcd", (3.0, 3.4, 3.8, 4.2), strict=True):
        start = pd.Timestamp("2023-01-01")
        for i in range(30):
            rows.append(
                dict(
                    athlete=a,
                    date=start + pd.Timedelta(days=7 * i),
                    distance_m=8000.0,
                    avg_speed_ms=speed,
                    is_race=False,
                    is_triathlon=0,
                )
            )
        for j in range(4):
            rows.append(
                dict(
                    athlete=a,
                    date=start + pd.Timedelta(days=215 + 28 * j),
                    distance_m=10000.0,
                    avg_speed_ms=speed * 1.1,
                    moving_s=10000 / (speed * 1.1),
                    is_race=True,
                    is_triathlon=0,
                )
            )
    ds = build_race_dataset(frame(rows))
    study = tune(ds, histgb_space, n_trials=3, n_splits=2)
    assert len(study.trials) == 3
    res = nested_evaluate(ds, histgb_space, n_trials=2, outer_splits=2, inner_splits=2)
    assert res["MAPE_%"] < 50
    assert RidgeBaseline  # imported for parity with the other test modules


tf = pytest.importorskip("tensorflow")


def test_pretrain_and_finetune_smoke(frame):
    from race_predictor.models.transfer import PretrainedEncoder, TransferGRU, pretrain_encoder

    enc = pretrain_encoder(build_marathon_data(kaggle_frame(8)), epochs=2, batch_size=4)
    rows = []
    for a in "ab":
        for i in range(12):
            rows.append(
                dict(
                    athlete=a,
                    date=pd.Timestamp("2023-01-01") + pd.Timedelta(days=7 * i),
                    is_race=False,
                    is_triathlon=0,
                )
            )
        for j in range(3):
            rows.append(
                dict(
                    athlete=a,
                    date=pd.Timestamp("2023-04-01") + pd.Timedelta(days=14 * j),
                    moving_s=3000.0,
                    is_race=True,
                    is_triathlon=0,
                )
            )
    ds = build_race_dataset(frame(rows))
    model = TransferGRU(pretrained=enc, freeze_epochs=1, epochs=2).fit(ds)
    assert model.predict(ds).shape == (len(ds),) and model.name == "GRU-finetuned"
    assert isinstance(enc, PretrainedEncoder)
    scratch = TransferGRU(epochs=2).fit(ds)
    assert np.isfinite(scratch.predict(ds)).all()
