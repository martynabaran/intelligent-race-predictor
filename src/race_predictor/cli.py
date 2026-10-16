"""Command line interface: ``race-predictor predict|evaluate|sources|app``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

from race_predictor.exceptions import RacePredictorError

app = typer.Typer(help="Race time prediction from Strava / Garmin exports.", no_args_is_help=True)
DataOpt = typer.Option(
    Path("data"), "--data", help="Folder with strava_activities/ and garmin_activities/."
)


@app.command()
def sources() -> None:
    """List supported export formats."""
    from race_predictor.sources import available_sources

    for name in available_sources():
        typer.echo(name)


@app.command()
def predict(
    export: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Strava/Garmin CSV export."
    ),
    data: Path = DataOpt,
    elevation: float = typer.Option(0.0, help="Race elevation gain in metres."),
    race_date: str = typer.Option(None, help="YYYY-MM-DD (default: day after the last activity)."),
) -> None:
    """Predict 5 km / 10 km / half / marathon times from an export."""
    from race_predictor.predictor import RacePredictor, format_time
    from race_predictor.sources import load_activities

    try:
        activities = load_activities(export)
        predictor = RacePredictor.from_data_dir(data)
        table = predictor.predict(
            activities,
            race_date=pd.Timestamp(race_date) if race_date else None,
            elevation_gain_m=elevation,
        )
    except RacePredictorError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from e

    typer.echo(f"{'Distance':<15}{'Predicted':>10}{'Range (80%)':>22}{'Riegel':>10}")
    for r in table.itertuples():
        rng = f"{format_time(r.low_s)} - {format_time(r.high_s)}"
        typer.echo(
            f"{r.distance:<15}{format_time(r.time_s):>10}{rng:>22}{format_time(r.riegel_s):>10}"
        )
    typer.echo(
        f"\nModel trained on {predictor.n_train_races_} races of {predictor.n_train_athletes_} "
        "athletes; the range is the empirical 10-90% error on unseen athletes."
    )


@app.command()
def evaluate(
    data: Path = DataOpt,
    with_gru: bool = typer.Option(False, help="Also train the GRU (needs tensorflow, slow)."),
) -> None:
    """Compare baseline models under chronological and athlete-grouped evaluation."""
    from race_predictor.evaluation.runner import compare_models, default_factories, load_dataset

    ds = load_dataset(data)
    typer.echo(f"{len(ds)} races from {ds.meta['athlete'].nunique()} athletes")
    typer.echo(compare_models(ds, default_factories(with_gru)).round(2).to_string(index=False))


@app.command("app")
def run_app(port: int = typer.Option(8501, help="Port for the Streamlit server.")) -> None:
    """Launch the Streamlit web app."""
    script = Path(__file__).parent / "app" / "streamlit_app.py"
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)]
        )
    )


if __name__ == "__main__":
    app()
