"""Small helpers."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timing:
    label: str
    seconds: float = 0.0


@contextmanager
def timer(label: str = "block") -> Iterator[Timing]:
    """Measure wall-clock time of a ``with`` block."""
    result = Timing(label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
