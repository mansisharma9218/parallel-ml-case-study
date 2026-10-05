"""Shared harness used by all three sub-problems (S01/S02/S03).

Change this package only via a PR all three teammates approve — it is
imported by everyone's experiment code, so an unreviewed change here can
silently break someone else's results.
"""

from .csv_schema import COLUMNS
from .csv_writer import append_row
from .environment import collect_environment, write_environment_json
from .memory import peak_memory_mb
from .timing import timer

__all__ = [
    "COLUMNS",
    "append_row",
    "collect_environment",
    "write_environment_json",
    "peak_memory_mb",
    "timer",
]
