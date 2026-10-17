"""Runs the full S03 experiment grid: for each problem size, the sequential
baseline, then both parallel strategies at each worker count. 1 warm-up +
5 timed runs per configuration. Resume-safe: re-running tops up missing
runs instead of duplicating rows.

Writes, under results/s03/:
  raw_results.csv   one shared-schema row per timed run
  stages.csv        per-stage wall/busy time for every timed run
  run_metrics.csv   per-run data-movement time and CPU utilization
  environment.json  hardware/software snapshot

Usage:
    python -m s03_preprocessing.experiment_runner            # full grid
    python -m s03_preprocessing.experiment_runner --quick    # tiny sanity run -> results/s03_quick
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

from common import append_row, timer, write_environment_json

from . import config
from .data_loading import ensure_raw_mnist
from .parallel import preprocess_parallel
from .resource_sampler import ResourceSampler
from .sequential import preprocess_sequential
from .stages import STAGES

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "s03"
QUICK_RESULTS_DIR = REPO_ROOT / "results" / "s03_quick"

KEY_COLUMNS = ["Dataset", "Problem_Size", "Workload", "Algorithm", "Resource_Count", "Run_ID"]
STAGES_COLUMNS = KEY_COLUMNS + ["Stage", "Wall_Time", "Busy_Time"]
RUN_METRICS_COLUMNS = KEY_COLUMNS + ["Data_Movement_Time", "CPU_Utilization_Pct"]

STRATEGY_ALGORITHM = {"stage": config.ALG_STAGE, "pipeline": config.ALG_PIPE}


def _append_csv(path, columns, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _hardware_label():
    system = platform.system()
    try:
        if system == "Darwin":
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2, check=True,
            ).stdout.strip()
            if brand:
                return brand
        elif system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{system}-{platform.machine()}"


def _load_completed_counts(csv_path):
    """(Dataset, Problem_Size, Workload, Algorithm, Resource_Count) -> rows already present."""
    counts = {}
    if not csv_path.exists():
        return counts
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["Dataset"], int(r["Problem_Size"]), r["Workload"], r["Algorithm"], int(r["Resource_Count"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def _record_run(out_dir, hardware, size, algorithm, workers, run_id, elapsed, sampler, result):
    sequential = algorithm == config.ALG_SEQ
    key = {
        "Dataset": config.DATASET,
        "Problem_Size": size,
        "Workload": config.WORKLOAD,
        "Algorithm": algorithm,
        "Resource_Count": workers,
        "Run_ID": run_id,
    }
    append_row(out_dir / "raw_results.csv", {
        **key,
        "Student_ID": config.STUDENT_ID,
        "Batch_ID": config.BATCH_ID,
        "Parallel_Model": "None" if sequential else config.PARALLEL_MODEL,
        "Hardware": hardware,
        "Threads": config.THREADS_PER_WORKER,
        "Execution_Time": elapsed,
        "Communication_Time": result.timing["communication"],
        "Synchronization_Time": result.timing["synchronization"],
        "Scheduling_Time": result.timing["scheduling"],
        "Memory_Use": round(sampler.peak_mb, 2),
        "Energy_if_available": "",
    })
    for stage in STAGES:
        _append_csv(out_dir / "stages.csv", STAGES_COLUMNS, {
            **key,
            "Stage": stage,
            "Wall_Time": result.stage_wall.get(stage, ""),
            "Busy_Time": result.stage_busy.get(stage, 0.0),
        })
    _append_csv(out_dir / "run_metrics.csv", RUN_METRICS_COLUMNS, {
        **key,
        "Data_Movement_Time": result.timing["data_movement"],
        "CPU_Utilization_Pct": sampler.cpu_percent,
    })


def _run_config(fn, out_dir, hardware, completed, size, algorithm, workers, n_warmup, n_timed):
    key = (config.DATASET, size, config.WORKLOAD, algorithm, workers)
    already_done = completed.get(key, 0)
    if already_done >= n_timed:
        print(f"  {algorithm} workers={workers}: already have {already_done}/{n_timed} runs, skipping")
        return
    print(f"  {algorithm} workers={workers}...", flush=True)
    for _ in range(n_warmup):
        fn()
    for run_id in range(already_done + 1, n_timed + 1):
        with ResourceSampler() as sampler, timer() as t:  # sampler start/stop stays outside the timed region
            result = fn()
        _record_run(out_dir, hardware, size, algorithm, workers, run_id, t.elapsed, sampler, result)


def run_grid(raw_path, out_dir, sizes, worker_counts, n_warmup, n_timed):
    completed = _load_completed_counts(out_dir / "raw_results.csv")
    if completed:
        print(f"Resuming: found existing results for {len(completed)} configuration(s).")
    hardware = _hardware_label()
    k = config.N_COMPONENTS

    for size in sizes:
        print(f"[{config.DATASET} n={size}]")
        _run_config(lambda: preprocess_sequential(raw_path, size, k),
                    out_dir, hardware, completed, size, config.ALG_SEQ, 1, n_warmup, n_timed)
        for workers in worker_counts:
            for strategy, algorithm in STRATEGY_ALGORITHM.items():
                _run_config(
                    lambda: preprocess_parallel(raw_path, size, k, workers, strategy, config.CHUNKS_PER_WORKER),
                    out_dir, hardware, completed, size, algorithm, workers, n_warmup, n_timed,
                )


def main():
    parser = argparse.ArgumentParser(description="Run the S03 preprocessing experiment grid.")
    parser.add_argument("--quick", action="store_true",
                        help="Tiny sizes/workers and 1 timed run, written to results/s03_quick.")
    args = parser.parse_args()

    if args.quick:
        out_dir = QUICK_RESULTS_DIR
        sizes, workers, n_timed = config.QUICK_PROBLEM_SIZES, config.QUICK_WORKER_COUNTS, config.QUICK_N_TIMED_RUNS
    else:
        if config.STUDENT_ID == "FILL_ME_IN":
            raise SystemExit("Set STUDENT_ID in s03_preprocessing/config.py before a full run.")
        out_dir = RESULTS_DIR
        sizes, workers, n_timed = config.PROBLEM_SIZES, config.WORKER_COUNTS, config.N_TIMED_RUNS

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        (out_dir / ".gitignore").write_text("*\n")  # never commit sanity-run output

    raw_path, _ = ensure_raw_mnist()
    env = write_environment_json(
        out_dir / "environment.json",
        extra={
            "student_id": config.STUDENT_ID,
            "batch_id": config.BATCH_ID,
            "cpu_model_detailed": _hardware_label(),
            "gpu": "none (CPU-only; no GPU used)",
            "optimization_flags": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
                "note": "BLAS pinned to 1 thread so the sequential baseline is genuinely single-threaded; see _pin_blas.py",
            },
            "runtime_configuration": {
                "problem_sizes": sizes,
                "worker_counts": workers,
                "n_components": config.N_COMPONENTS,
                "chunks_per_worker": config.CHUNKS_PER_WORKER,
                "n_warmup_runs": config.N_WARMUP_RUNS,
                "n_timed_runs": n_timed,
                "start_method": "spawn",
            },
        },
    )
    print("Environment recorded:", env)

    run_grid(raw_path, out_dir, sizes, workers, config.N_WARMUP_RUNS, n_timed)
    print(f"Done. Raw results appended to {out_dir / 'raw_results.csv'}")


if __name__ == "__main__":
    main()
