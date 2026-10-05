"""Runs the full S01 experiment grid: for each dataset x problem size x K,
times the sequential baseline once, then the parallel version at each
worker count. 1 warm-up + 5 timed runs per configuration, matching the
spec. Writes raw rows (one per timed run) to results/s01/raw_results.csv
and the environment snapshot once to results/s01/environment.json.

Usage:
    python3 -m s01_kmeans.experiment_runner            # full grid
    python3 -m s01_kmeans.experiment_runner --quick     # tiny sanity run
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import argparse
import csv
import platform
from pathlib import Path

from common import append_row, peak_memory_mb, timer, write_environment_json

from . import config
from .data_loading import generate_blobs, load_mnist
from .parallel import kmeans_parallel
from .sequential import init_centroids, kmeans_sequential

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "s01"
RAW_CSV_PATH = RESULTS_DIR / "raw_results.csv"
ITERATIONS_CSV_PATH = RESULTS_DIR / "iterations.csv"
ENV_JSON_PATH = RESULTS_DIR / "environment.json"

ITERATIONS_COLUMNS = [
    "Dataset",
    "Problem_Size",
    "K",
    "Algorithm",
    "Resource_Count",
    "Run_ID",
    "Iterations",
]


def _append_iterations_row(row):
    ITERATIONS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = ITERATIONS_CSV_PATH.exists()
    with open(ITERATIONS_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ITERATIONS_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _hardware_label():
    return f"{platform.system()}-{platform.machine()}"


def _base_row(workload, dataset, size, algorithm, parallel_model, resource_count, threads, run_id):
    return {
        "Student_ID": config.STUDENT_ID,
        "Batch_ID": config.BATCH_ID,
        "Workload": workload,
        "Dataset": dataset,
        "Problem_Size": size,
        "Algorithm": algorithm,
        "Parallel_Model": parallel_model,
        "Hardware": _hardware_label(),
        "Resource_Count": resource_count,
        "Threads": threads,
        "Run_ID": run_id,
    }


def _run_sequential_config(X, k, dataset_name, size, n_warmup, n_timed):
    initial = init_centroids(X, k, seed=config.RANDOM_SEED)

    for _ in range(n_warmup):
        kmeans_sequential(X, k, initial, max_iter=config.MAX_ITER, tol=config.TOLERANCE)

    for run_id in range(1, n_timed + 1):
        with timer() as t:
            _, _, n_iter = kmeans_sequential(
                X, k, initial, max_iter=config.MAX_ITER, tol=config.TOLERANCE
            )
        row = _base_row(
            workload=f"kmeans_k{k}",
            dataset=dataset_name,
            size=X.shape[0],
            algorithm="Lloyd-Sequential",
            parallel_model="None",
            resource_count=1,
            threads=1,
            run_id=run_id,
        )
        row["Execution_Time"] = t.elapsed
        row["Communication_Time"] = 0.0
        row["Synchronization_Time"] = 0.0
        row["Scheduling_Time"] = 0.0
        row["Memory_Use"] = peak_memory_mb()
        row["Energy_if_available"] = ""
        append_row(RAW_CSV_PATH, row)
        _append_iterations_row(
            {
                "Dataset": dataset_name,
                "Problem_Size": X.shape[0],
                "K": k,
                "Algorithm": "Lloyd-Sequential",
                "Resource_Count": 1,
                "Run_ID": run_id,
                "Iterations": n_iter,
            }
        )


def _run_parallel_config(X, k, n_workers, dataset_name, size, n_warmup, n_timed):
    initial = init_centroids(X, k, seed=config.RANDOM_SEED)

    for _ in range(n_warmup):
        kmeans_parallel(
            X, k, initial, n_workers=n_workers, max_iter=config.MAX_ITER, tol=config.TOLERANCE
        )

    for run_id in range(1, n_timed + 1):
        with timer() as t:
            result = kmeans_parallel(
                X, k, initial, n_workers=n_workers, max_iter=config.MAX_ITER, tol=config.TOLERANCE
            )
        row = _base_row(
            workload=f"kmeans_k{k}",
            dataset=dataset_name,
            size=X.shape[0],
            algorithm="Lloyd-Parallel-Pool",
            parallel_model="multiprocessing.Pool+shared_memory",
            resource_count=n_workers,
            threads=1,
            run_id=run_id,
        )
        row["Execution_Time"] = t.elapsed
        row["Communication_Time"] = result.timing["communication"]
        row["Synchronization_Time"] = result.timing["synchronization"]
        row["Scheduling_Time"] = result.timing["scheduling"]
        row["Memory_Use"] = peak_memory_mb()
        row["Energy_if_available"] = ""
        append_row(RAW_CSV_PATH, row)
        _append_iterations_row(
            {
                "Dataset": dataset_name,
                "Problem_Size": X.shape[0],
                "K": k,
                "Algorithm": "Lloyd-Parallel-Pool",
                "Resource_Count": n_workers,
                "Run_ID": run_id,
                "Iterations": result.n_iterations,
            }
        )


def run_grid(blob_sizes, k_values, worker_counts, n_warmup, n_timed, include_mnist=True):
    for size in blob_sizes:
        X = generate_blobs(
            n_samples=size,
            n_features=config.BLOB_N_FEATURES,
            centers=config.BLOB_TRUE_CENTERS,
            seed=config.RANDOM_SEED,
        )
        for k in k_values:
            print(f"[blobs n={size} k={k}] sequential baseline...")
            _run_sequential_config(X, k, "synthetic_blobs", size, n_warmup, n_timed)
            for n_workers in worker_counts:
                print(f"[blobs n={size} k={k}] parallel workers={n_workers}...")
                _run_parallel_config(X, k, n_workers, "synthetic_blobs", size, n_warmup, n_timed)

    if include_mnist:
        X, source = load_mnist()
        print(f"[mnist n={X.shape[0]}] (source={source})")
        for k in k_values:
            print(f"[mnist k={k}] sequential baseline...")
            _run_sequential_config(X, k, "mnist", X.shape[0], n_warmup, n_timed)
            for n_workers in worker_counts:
                print(f"[mnist k={k}] parallel workers={n_workers}...")
                _run_parallel_config(X, k, n_workers, "mnist", X.shape[0], n_warmup, n_timed)


def main():
    parser = argparse.ArgumentParser(description="Run the S01 K-means experiment grid.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny sizes/K/workers and 1 timed run, to sanity-check the pipeline in under a minute.",
    )
    parser.add_argument(
        "--no-mnist",
        action="store_true",
        help="Skip MNIST (e.g. no network access yet for the fallback download).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env = write_environment_json(ENV_JSON_PATH, extra={"student_id": config.STUDENT_ID, "batch_id": config.BATCH_ID})
    print("Environment recorded:", env)

    if args.quick:
        run_grid(
            blob_sizes=[2_000, 5_000],
            k_values=[2, 4],
            worker_counts=[1, 2],
            n_warmup=1,
            n_timed=1,
            include_mnist=not args.no_mnist,
        )
    else:
        run_grid(
            blob_sizes=config.BLOB_SIZES,
            k_values=config.K_VALUES,
            worker_counts=config.WORKER_COUNTS,
            n_warmup=config.N_WARMUP_RUNS,
            n_timed=config.N_TIMED_RUNS,
            include_mnist=not args.no_mnist,
        )

    print(f"Done. Raw results appended to {RAW_CSV_PATH}")


if __name__ == "__main__":
    main()
