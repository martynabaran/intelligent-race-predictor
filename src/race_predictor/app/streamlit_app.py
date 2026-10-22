"""Streamlit app: upload a Strava/Garmin export, get predicted race times.

Run with ``race-predictor app`` or ``streamlit run src/race_predictor/app/streamlit_app.py``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from race_predictor.domain import RaceMode
from race_predictor.exceptions import RacePredictorError
from race_predictor.services import DISTANCES_M, RacePredictor, format_time, load_activities

DATA_DIR = Path(os.environ.get("RACE_PREDICTOR_DATA", "data"))
MODEL_LABELS = {
    "Riegel": "the Riegel formula",
    "Riegel+correction": "the Riegel formula with a learned correction",
    "Riegel+correction+ref": "the Riegel formula with a learned correction",
    "Ridge": "ridge regression",
}


@st.cache_resource(show_spinner="Training the model on the bundled race data...")
def get_predictor() -> RacePredictor:
    """Train (once per server process) the predictor on the bundled race data."""
    return RacePredictor.from_data_dir(DATA_DIR)


def sample_files() -> dict[str, Path]:
    """Map a display label to each anonymised sample export under the data folder."""
    files = sorted(DATA_DIR.glob("*_activities/*.csv"))
    return {f"{f.parent.name.split('_')[0]} / {f.stem}": f for f in files}


def load_upload(upload: UploadedFile) -> pd.DataFrame:
    """Parse an uploaded Strava/Garmin CSV into the canonical activity table."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(upload.getvalue())
    try:
        return load_activities(tmp.name, athlete_id="you")
    finally:
        Path(tmp.name).unlink()


def explain_riegel(
    predictor: RacePredictor, activities: pd.DataFrame, contrib: pd.Series, label: str
) -> None:
    """Explain a prediction as the Riegel projection of the reference run (+ the correction)."""
    ref = predictor.reference_run(activities)
    km = DISTANCES_M[label] / 1000
    riegel = (contrib["best_run_pace"] + contrib.get("riegel_distance_adjustment", 0.0)) * km
    correction = contrib.get("model_correction", 0.0) * km
    kind = "a race" if ref["is_race"] else "a training run, not a race"
    st.markdown(
        f"Your fastest run of 5 km or more in the last year: **{ref['distance_m'] / 1000:.1f} km "
        f"in {format_time(ref['time_s'])}** ({format_time(ref['pace_s_km'])} per km) - "
        f"{kind}.\n\n"
        f"- Riegel formula, T2 = T1·(D2/D1)^1.06, projects it to {label}: "
        f"**{format_time(riegel)}**\n"
        f"- Learned correction: **{'+' if correction >= 0 else '−'}"
        f"{format_time(abs(correction))}**\n"
        f"- Prediction: **{format_time(riegel + correction)}**"
    )
    st.caption(
        "The correction is learned from past races: Riegel tends to be optimistic for long "
        "races and when it projects a training run rather than a race result, and the model "
        "knows which one yours is. On athletes it had not seen, this was the most accurate "
        "model (median error ~5%)."
    )


@dataclass(frozen=True)
class SidebarChoices:
    """What the sidebar offers and what the user picked in it."""

    upload: UploadedFile | None
    sample: str  # key of ``samples``, or "-" when none is chosen
    samples: dict[str, Path]
    elevation_m: float
    predictor: RacePredictor


def render_sidebar(mode: RaceMode) -> SidebarChoices:
    """Draw the sidebar and return the user's choices."""
    with st.sidebar:
        st.header("Your data")
        upload = st.file_uploader("Strava `activities.csv` or Garmin Connect CSV", type="csv")
        samples = sample_files()
        sample = st.selectbox("...or try an anonymised sample", ["-", *samples])
        elevation = st.number_input("Race elevation gain (m)", 0, 3000, 0, step=50)
        st.divider()
        predictor = get_predictor()
        model = predictor.model_name(mode)
        st.caption(
            f"Model: {MODEL_LABELS.get(model, model)}, chosen by evaluation on "
            f"{predictor.n_train_races_} running races of {predictor.n_train_athletes_} athletes. "
            "Ranges are the empirical 10-90% errors on athletes the model had not seen."
        )
    return SidebarChoices(upload, sample, samples, float(elevation), predictor)


def load_chosen_activities(choices: SidebarChoices) -> pd.DataFrame | None:
    """Load the uploaded export or chosen sample; ``None`` (with a hint) if neither is given."""
    if choices.upload is not None:
        return load_upload(choices.upload)
    if choices.sample != "-":
        return load_activities(choices.samples[choices.sample], athlete_id="you")
    st.info("Upload an export (or pick a sample) in the sidebar to get started.")
    return None


def render_predictions(activities: pd.DataFrame, table: pd.DataFrame) -> None:
    """Show the activity summary, one metric per distance and the predictions table."""
    runs = activities[activities["type"] == "Run"]
    st.subheader("Predicted times if you raced today")
    st.caption(
        f"{len(activities)} activities ({len(runs)} runs), "
        f"{activities['date'].min():%Y-%m-%d} to {activities['date'].max():%Y-%m-%d}"
    )
    cols = st.columns(len(table))
    for col, row in zip(cols, table.to_dict("records"), strict=True):
        delta = (
            None
            if pd.isna(row["riegel_s"])
            else f"{(row['time_s'] - row['riegel_s']) / 60:+.1f} min vs Riegel"
        )
        col.metric(row["distance"], format_time(row["time_s"]), delta, delta_color="off")
        col.caption(f"{format_time(row['low_s'])} - {format_time(row['high_s'])}")

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


def render_form_curve(predictor: RacePredictor, activities: pd.DataFrame) -> str:
    """Draw the form-over-time chart; return the distance label chosen for it."""
    st.subheader("Form over time")
    label = st.selectbox("Distance", list(DISTANCES_M), index=1)
    curve = predictor.form_curve(activities, DISTANCES_M[label])
    if curve.empty:
        st.info("Not enough history for a form curve (needs > 90 days of runs).")
        return label
    fig = go.Figure(go.Scatter(x=curve["date"], y=curve["time_s"] / 60, mode="lines+markers"))
    fig.update_layout(
        yaxis_title=f"Predicted {label} time (min)",
        xaxis_title=None,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower is better: the time the model would have predicted on each date.")
    return label


def render_drivers(
    predictor: RacePredictor,
    activities: pd.DataFrame,
    label: str,
    elevation: float,
    mode: RaceMode,
) -> None:
    """Explain what drives the prediction for ``label``: Riegel text or a contribution chart."""
    st.subheader("What drives the prediction")
    contrib = predictor.explain(
        activities, DISTANCES_M[label], elevation_gain_m=elevation, mode=mode
    )
    if "best_run_pace" in contrib:
        explain_riegel(predictor, activities, contrib, label)
        return
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
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = faster, red = slower than the average training race (exact, linear model).")


def main() -> None:
    """Render the main page."""
    st.set_page_config(page_title="Race Performance Predictor", page_icon="🏃", layout="wide")
    st.title("🏃 Race Performance Predictor")
    st.caption(
        "Predicts 5 km - marathon times from your training history. "
        "Extension of the thesis *Race Performance Predictor* by Zuzanna Jarlaczyńska."
    )

    mode = RaceMode.ROAD  # running races only (no data for triathlon runs yet)
    choices = render_sidebar(mode)

    try:
        activities = load_chosen_activities(choices)
        if activities is None:
            return
        st.session_state["activities"] = activities  # shared with the Training load page
        table = choices.predictor.predict(
            activities, elevation_gain_m=choices.elevation_m, mode=mode
        )
    except RacePredictorError as e:
        st.error(str(e))
        return

    render_predictions(activities, table)
    left, right = st.columns(2)
    with left:
        label = render_form_curve(choices.predictor, activities)
    with right:
        render_drivers(choices.predictor, activities, label, choices.elevation_m, mode)


main()
