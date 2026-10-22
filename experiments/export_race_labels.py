"""Turn the filled-in review (docs/REVIEW_RACES.md) into data/race_labels.csv.

Run once after reviewing; the CSV is then the source of truth and `review_races.py` pre-fills the
decision column from it, so regenerating the review lists no longer wipes the decisions.

    python experiments/export_race_labels.py
"""

import argparse
import re
from pathlib import Path

import pandas as pd

DECISIONS = {"tak": True, "nie": False}  # anything else ("?", blank) is not a decision
SECTION = re.compile(r"^## ([ABC])\.")


def parse_review(text: str) -> pd.DataFrame:
    """Rows of the three review tables that carry a decision: athlete, date, km, list, decision."""
    rows, section = [], None
    for line in text.splitlines():
        if match := SECTION.match(line):
            section = match.group(1)
        if section is None or not line.startswith("|") or "athlete" in line or ":--" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        decision = DECISIONS.get(cells[-1].lower())
        if decision is not None:
            rows.append(
                {
                    "athlete": cells[0],
                    "date": cells[1],
                    "distance_km": float(cells[2]),
                    "is_race": decision,
                    "list": section,
                }
            )
    return pd.DataFrame(rows, columns=["athlete", "date", "distance_km", "is_race", "list"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=Path("docs/REVIEW_RACES.md"))
    ap.add_argument("--out", type=Path, default=Path("data/race_labels.csv"))
    args = ap.parse_args()

    labels = parse_review(args.review.read_text()).sort_values(["athlete", "date", "distance_km"])
    labels["is_race"] = labels["is_race"].astype(int)
    labels.to_csv(args.out, index=False)
    print(labels.groupby(["list", "is_race"]).size().unstack(fill_value=0))
    print(f"wrote {len(labels)} labels to {args.out}")


if __name__ == "__main__":
    main()
