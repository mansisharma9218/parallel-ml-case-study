"""Identifiers and fixed experiment constants shared across s01_kmeans
modules, so every raw CSV row and environment record is consistent.
"""

from pathlib import Path
from typing import NamedTuple

STUDENT_ID = "23BCE0856"
BATCH_ID = "CS10"

RANDOM_SEED = 42

BLOB_SIZES = [100_000, 500_000, 2_000_000]
BLOB_N_FEATURES = 32
BLOB_TRUE_CENTERS = 10

K_VALUES = [8, 16, 64]
WORKER_COUNTS = [1, 2, 4, 8]

N_WARMUP_RUNS = 1
N_TIMED_RUNS = 5

MAX_ITER = 100
TOLERANCE = 1e-4

REPO_ROOT = Path(__file__).resolve().parent.parent


class ResultPaths(NamedTuple):
    results_dir: Path
    raw_csv: Path
    iterations_csv: Path
    summary_csv: Path
    environment_json: Path
    plots_dir: Path


def results_paths(quick=False):
    """Output locations; --quick runs go to a separate, git-ignored folder."""
    d = REPO_ROOT / "results" / ("s01_quick" if quick else "s01")
    return ResultPaths(
        d, d / "raw_results.csv", d / "iterations.csv",
        d / "summary.csv", d / "environment.json", d / "plots",
    )
