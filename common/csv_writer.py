"""Appends result rows to a shared-schema raw-results CSV, writing the
header once. Speedup/Efficiency are computed later by analysis code from
raw times, never typed in here.
"""

import csv
from pathlib import Path

from .csv_schema import COLUMNS


def append_row(csv_path, row):
    """row: dict keyed by (a subset of) COLUMNS. Missing keys are written
    empty — raw CSVs are expected to leave Speedup/Efficiency blank.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in COLUMNS})


def append_rows(csv_path, rows):
    for row in rows:
        append_row(csv_path, row)
