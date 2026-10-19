"""Streamlit app: upload a Strava/Garmin export, get predicted race times.

Run with ``race-predictor app`` or ``streamlit run src/race_predictor/app/streamlit_app.py``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from race_predictor.domain import RaceMode
from race_predictor.exceptions import RacePredictorError
from race_predictor.predictor import DISTANCES_M, RacePredictor, format_time
from race_predictor.sources import load_activities

DATA_DIR = Path(os.environ.get("RACE_PREDICTOR_DATA", "data"))
MODES = {"Road race": RaceMode.ROAD, "Triathlon (run leg)": RaceMode.TRIATHLON}
MODEL_LABELS = {"Riegel": "the Riegel formula", "Ridge": "ridge regression"}


@st.cache_resource(show_spinner="Training the model on the bundled race data...")
def get_predictor() -> RacePredictor:
    return RacePredictor.from_data_dir(DATA_DIR)


def sample_files() -> dict[str, Path]:
    files = sorted(DATA_DIR.glob("*_activities/*.csv"))
    return {f"{f.parent.name.split('_')[0]} / {f.stem}": f for f in files}


def load_upload(upload) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(upload.getvalue())
    try:
        return load_activities(tmp.name, athlete_id="you")
    finally:
        os.unlink(tmp.name)


def main() -> None:
    st.set_page_config(page_title="Race Performance Predictor", page_icon="🏃", layout="wide")
    st.title("🏃 Race Performance Predictor")
    st.caption(
        "Predicts 5 km - marathon times from your training history. "
        "Extension of the thesis *Race Performance Predictor* by Zuzanna Jarlaczyńska."
    )

    with st.sidebar:
        st.header("Your data")
        upload = st.file_uploader("Strava `activities.csv` or Garmin Connect CSV", type="csv")
        samples = sample_files()
        sample = st.selectbox("...or try an anonymised sample", ["-"] + list(samples))
        mode = MODES[st.radio("Race type", list(MODES))]
        elevation = st.number_input("Race elevation gain (m)", 0, 3000, 0, step=50)
        st.divider()
        predictor = get_predictor()
        model = predictor.model_name(mode)
        st.caption(
            f"Model for this race type: {MODEL_LABELS.get(model, model)}, chosen by evaluation "
            f"on {predictor.n_train_races_} races of {predictor.n_train_athletes_} athletes. "
            "Ranges are the empirical 10-90% errors of this race type on athletes the model "
            "had not seen."
        )

    try:
        if upload is not None:
            activities = load_upload(upload)
        elif sample != "-":
            activities = load_activities(samples[sample], athlete_id="you")
        else:
            st.info("Upload an export (or pick a sample) in the sidebar to get started.")
            return
        table = predictor.predict(activities, elevation_gain_m=float(elevation), mode=mode)
    except RacePredictorError as e:
        st.error(str(e))
        return

    runs = activities[activities["type"] == "Run"]
    st.subheader("Predicted times if you raced today")
    st.caption(
        f"{len(activities)} activities ({len(runs)} runs), "
        f"{activities['date'].min():%Y-%m-%d} to {activities['date'].max():%Y-%m-%d}"
    )
    cols = st.columns(len(table))
    for col, r in zip(cols, table.itertuples(), strict=True):
        delta = (
            None if pd.isna(r.riegel_s) else f"{(r.time_s - r.riegel_s) / 60:+.1f} min vs Riegel"
        )
        col.metric(r.distance, format_time(r.time_s), delta, delta_color="off")
        col.caption(f"{format_time(r.low_s)} - {format_time(r.high_s)}")

    shown = table.assign(
        Predicted=table["time_s"].map(format_time),
        **{
            "Range (10-90%)": table["low_s"].map(format_time)
            + " - "
            + table["high_s"].map(format_time)
        },
        Riegel=table["riegel_s"].map(format_time),
    )[["distance", "Predicted", "Range (10-90%)", "Riegel"]]
    st.dataframe(shown, hide_index=True, use_container_width=True)
    st.caption(
        "Riegel = your fastest 5 km+ training run projected with T2 = T1·(D2/D1)^1.06; it ignores "
        "training volume and tends to be optimistic because training runs are rarely race efforts."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Form over time")
        label = st.selectbox("Distance", list(DISTANCES_M), index=1)
        curve = predictor.form_curve(activities, DISTANCES_M[label])
        if curve.empty:
            st.info("Not enough history for a form curve (needs > 90 days of runs).")
        else:
            fig = go.Figure(
                go.Scatter(x=curve["date"], y=curve["time_s"] / 60, mode="lines+markers")
            )
            fig.update_layout(
                yaxis_title=f"Predicted {label} time (min)",
                xaxis_title=None,
                margin=dict(l=10, r=10, t=10, b=10),
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Lower is better: the time the model would have predicted on each date.")
    with right:
        st.subheader("What drives the prediction")
        if model == "Riegel":
            ref = predictor.reference_run(activities)
            km = ref["distance_m"] / 1000
            st.markdown(
                f"Your fastest run of 5 km or more in the last year: **{km:.1f} km in "
                f"{format_time(ref['time_s'])}** ({format_time(ref['pace_s_km'])} per km). "
                "For road races the Riegel formula projects it to each distance, "
                "T2 = T1·(D2/D1)^1.06. On athletes it had not seen it was the most accurate "
                "model for road races (median error ~5%)."
            )
            return
        contrib = predictor.explain(
            activities, DISTANCES_M[label], elevation_gain_m=float(elevation), mode=mode
        )
        top = contrib.drop("(baseline)").head(8)[::-1]
        fig = go.Figure(
            go.Bar(
                x=top.values,
                y=top.index,
                orientation="h",
                marker_color=["#d62728" if v > 0 else "#2ca02c" for v in top.values],
            )
        )
        fig.update_layout(
            xaxis_title="effect on pace (s/km)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Green = faster, red = slower than the average training race (exact, linear model)."
        )


main()
