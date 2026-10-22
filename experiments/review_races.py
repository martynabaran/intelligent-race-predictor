"""Lists of detected races to review by hand (stage 8c). Writes docs/REVIEW_RACES.md.

A. Races labelled only because they follow a ride and have a standard distance (score 1 + 1):
   possibly brick training sessions rather than triathlon races.
B. Races that appear only with the HR-spike filter (realistic HRmax lowers the HR threshold).
C. Stand-alone races whose distance is not within ±10 % of 5 k / 10 k / half / marathon.

Only races that enter the model data (>= 3 runs in the preceding year) are listed.

python experiments/review_races.py
"""

import argparse
from pathlib import Path

import pandas as pd

from race_predictor.features import build_race_dataset
from race_predictor.processing import HeuristicRaceDetector, default_pipeline, distance_label
from race_predictor.sources import load_directory

REASONS = ["high_hr", "speed_spike", "ultra", "after_ride", "standard_distance"]
KEY = ["athlete", "date", "distance_m"]


def labelled(raw: pd.DataFrame, pipeline) -> pd.DataFrame:
    """Races entering the model data, with the detector's reasons."""
    df = pipeline.run(raw)
    df = df.join(HeuristicRaceDetector().reasons(df).drop(columns="is_race"))
    meta = build_race_dataset(df).meta.rename(
        columns={"race_date": "date", "race_distance_m": "distance_m"}
    )
    return df.merge(meta, on=KEY)


def pace(speed_ms: float) -> str:
    s = round(1000 / speed_ms)
    return f"{s // 60}:{s % 60:02d}"


def decisions_from(path: Path) -> dict[tuple[str, str, float], str]:
    """Decisions already recorded in the labels CSV, keyed like the review rows."""
    if not path.exists():
        return {}
    labels = pd.read_csv(path)
    return {
        (r.athlete, r.date, round(r.distance_km, 2)): "tak" if r.is_race else "nie"
        for r in labels.itertuples()
    }


def table(
    races: pd.DataFrame,
    hrmax_before: pd.Series | None = None,
    decisions: dict[tuple[str, str, float], str] | None = None,
) -> str:
    if races.empty:
        return "_(brak)_\n"
    out = pd.DataFrame(
        {
            "athlete": races["athlete"],
            "date": races["date"].dt.strftime("%Y-%m-%d"),
            "km": (races["distance_m"] / 1000).round(2),
            "pace /km": races["avg_speed_ms"].map(pace),
            "avg HR": races["avg_hr"].round(0),
            "HRmax": races["hr_max"].round(0),
            "% HRmax": (100 * races["avg_hr"] / races["hr_max"]).round(0),
            "after ride": races["is_triathlon"].map({1: "tak", 0: ""}),
            "reasons": races[REASONS].apply(lambda r: ", ".join(r.index[r.astype(bool)]), axis=1),
            "decyzja (start?)": "",
        }
    )
    keys = zip(out["athlete"], out["date"], out["km"], strict=True)
    out["decyzja (start?)"] = [(decisions or {}).get((a, d, k), "") for a, d, k in keys]
    if hrmax_before is not None:
        out.insert(5, "HRmax before", races["athlete"].map(hrmax_before).round(0))
    return out.sort_values(["athlete", "date"]).to_markdown(index=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("docs/REVIEW_RACES.md"))
    args = ap.parse_args()

    raw = pd.concat(
        [
            load_directory(args.data / "strava_activities"),
            load_directory(args.data / "garmin_activities"),
        ],
        ignore_index=True,
    )
    decisions = decisions_from(args.data / "race_labels.csv")
    base = labelled(raw, default_pipeline(hr_spike_filter=False, include_triathlon_races=True))
    spikes = labelled(raw, default_pipeline(hr_spike_filter=True, include_triathlon_races=True))

    only_brick = base[
        base["after_ride"]
        & base["standard_distance"]
        & ~base[["high_hr", "speed_spike", "ultra"]].any(axis=1)
    ]
    added = spikes.merge(base[KEY], on=KEY, how="left", indicator=True)
    added = added[added["_merge"] == "left_only"].drop(columns="_merge")
    removed = base.merge(spikes[KEY], on=KEY, how="left", indicator=True)
    removed = removed[removed["_merge"] == "left_only"]
    odd = base[(base["is_triathlon"] == 0) & (base["distance_m"].map(distance_label) == "other")]
    hrmax_before = base.groupby("athlete")["hr_max"].first()

    summary = (
        f"Dane modelu: **{len(base)} startów** "
        f"({int(base['is_triathlon'].sum())} po rowerze).\n\n"
        f"- **A.** Starty wykryte tylko dlatego, że są po rowerze i mają standardowy dystans: "
        f"**{len(only_brick)}** z {int(base['is_triathlon'].sum())} startów po rowerze.\n"
        f"- **B.** Starty dochodzące po włączeniu filtra skoków tętna: **{len(added)}** "
        f"(znikające: {len(removed)}).\n"
        f"- **C.** Zwykłe starty o niestandardowym dystansie: **{len(odd)}**.\n"
    )
    print(summary)
    args.out.write_text(
        "# Przegląd wykrytych startów\n\n"
        "Wygenerowane przez `experiments/review_races.py`. W pracy startów nie ma w danych "
        "jako etykiet - wykrywa je heurystyka `HeuristicRaceDetector` (punkty: wysokie tętno +2, "
        "skok prędkości vs średnia z roku +2, >40 km +2, bieg po rowerze +1, standardowy dystans +1; "
        "start = co najmniej 2 punkty). Proszę o wpisanie w kolumnie *decyzja* `tak` / `nie` / `?` "
        "(znasz te dane najlepiej). Decyzje są zapisane w `data/race_labels.csv` (eksport: "
        "`experiments/export_race_labels.py`) i wpisywane tu automatycznie.\n\n"
        + summary
        + "\n## A. Tylko „po rowerze” + standardowy dystans (może trening typu brick?)\n\n"
        "Bez wysokiego tętna i bez skoku prędkości. Jeśli to w większości treningi, a nie zawody, "
        "trzeba podnieść próg albo wymagać wysokiego tętna dla biegów po rowerze.\n\n"
        + table(only_brick, decisions=decisions)
        + "\n## B. Dochodzą po filtrze skoków tętna (realne HRmax)\n\n"
        "`HRmax before` = HRmax zawyżone przez skoki czujnika; `HRmax` = po filtrze. Niższe HRmax "
        "obniża próg tętna, więc te biegi zaczynają spełniać warunek wysokiego tętna.\n\n"
        + table(added, hrmax_before, decisions)
        + "\n## C. Zwykłe starty o niestandardowym dystansie\n\n"
        "Np. biegi na 7 km, 15 km, trail lub niedokładny GPS. Czy to zawody?\n\n"
        + table(odd, decisions=decisions)
    )


if __name__ == "__main__":
    main()
