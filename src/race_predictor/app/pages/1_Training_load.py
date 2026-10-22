"""Streamlit page: fitness, fatigue and form over time (uses the data chosen on the main page)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from race_predictor.exceptions import RacePredictorError
from race_predictor.services import training_load

st.set_page_config(page_title="Training load", page_icon="📈", layout="wide")
st.title("📈 Training load")
st.caption(
    "Fitness (CTL, 42-day average load), fatigue (ATL, 7-day) and form (TSB = CTL - ATL), "
    "computed from every activity with the Banister TRIMP load."
)

activities = st.session_state.get("activities")
if activities is None:
    st.info("Pick an export (or a sample) on the main page first.")
    st.stop()

try:
    state, summary = training_load(activities)
except RacePredictorError as e:
    st.error(str(e))
    st.stop()

fitness, fatigue, form = st.columns(3)
fitness.metric("Fitness (CTL)", f"{summary.ctl:.0f}")
fatigue.metric("Fatigue (ATL)", f"{summary.atl:.0f}")
form.metric("Form (TSB)", f"{summary.tsb:+.0f}", summary.form, delta_color="off")

window_days = st.slider("Days shown", 60, min(len(state), 1500), min(len(state), 365), step=30)
shown = state.iloc[-window_days:]

fig = go.Figure()
fig.add_scatter(x=shown.index, y=shown["ctl"], name="Fitness (CTL)", line={"color": "#1f77b4"})
fig.add_scatter(x=shown.index, y=shown["atl"], name="Fatigue (ATL)", line={"color": "#d62728"})
fig.add_bar(x=shown.index, y=shown["tsb"], name="Form (TSB)", marker_color="rgba(120,120,120,.45)")
fig.update_layout(
    height=420, margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h"}
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"{summary.estimated_share:.0%} of the load comes from activities without usable heart rate, "
    "priced at an assumed typical intensity for the activity type. "
    "The first ~6 weeks of a history underestimate fitness (the averages start from zero). "
    "Form labels are a rule of thumb: fresh above +5, fatigued below -10."
)
