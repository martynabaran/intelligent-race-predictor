"""Writing result reports whose numbers come from scripts but whose conclusions are hand-written."""

from __future__ import annotations

from pathlib import Path

CONCLUSIONS = "## Conclusions"


def write_report(path: Path, generated: str) -> None:
    """Write ``generated`` to ``path``, keeping its hand-written ``## Conclusions`` section.

    Numbers come from the scripts; the conclusions already in the file are edited by hand.
    """
    previous = path.read_text() if path.exists() else ""
    if CONCLUSIONS in previous:
        kept = previous[previous.index(CONCLUSIONS) :]
    else:
        kept = f"{CONCLUSIONS}\n\n(to be written after reading the tables above)\n"
    path.write_text(generated.rstrip("\n") + "\n\n" + kept)
