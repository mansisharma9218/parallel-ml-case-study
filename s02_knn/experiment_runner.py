"""Runs the full dataset x train-size x query-count x workers grid:
1 warm-up + 5 timed runs per configuration, resume-safe.

    python3 -m s02_knn.experiment_runner --quick   # tiny offline sanity run
    python3 -m s02_knn.experiment_runner           # full MNIST run
"""
from . import _pin_blas  # noqa: F401
import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np

from common.csv_writer import append_row
from common.environment import write_environment_json
from common.timing import timer

from . import config as C
from .data_loading import load_digits, load_mnist, split_train_query
from .knn import PoolRunner, SearchContext, search_sequential, vote
from .memory_sampler import MemorySampler

ACC_COLUMNS = ["Dataset", "Problem_Size", "Workload", "Algorithm",
               "Resource_Count", "Run_ID", "Accuracy"]


def _sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def hardware_name():
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.machine()
    return platform.processor() or platform.machine()


def count_existing(raw_csv, dataset, n_train, workload, algo, p):
    if not Path(raw_csv).exists():
        return 0
    n = 0
    with open(raw_csv, newline="") as f:
        for r in csv.DictReader(f):
            if (r["Dataset"] == dataset and r["Problem_Size"] == str(n_train)
                    and r["Workload"] == workload and r["Algorithm"] == algo
                    and r["Resource_Count"] == str(p)):
                n += 1
    return n


def append_accuracy(acc_csv, row):
    acc_csv = Path(acc_csv)
    new = not acc_csv.exists()
    with open(acc_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ACC_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def run_series(fn, need, start_id, dataset, n_train, n_query, algo, p, sched,
               hardware, ytr, yq, raw_csv, acc_csv):
    """fn() -> (neighbour_idx, comm, sync). Writes `need` timed rows."""
    workload = f"knn_k{C.K_NEIGHBORS}_q{n_query}"
    for _ in range(C.N_WARMUP):
        fn()
    for i in range(need):
        run_id = start_id + i
        with MemorySampler() as ms:
            with timer() as t:
                idx, comm, sync = fn()
        # accuracy is computed OUTSIDE the timed region
        acc = float((vote(ytr[idx]) == yq).mean())
        seq = algo == C.ALG_SEQ
        append_row(raw_csv, {
            "Student_ID": C.STUDENT_ID, "Batch_ID": C.BATCH_ID,
            "Workload": workload, "Dataset": dataset, "Problem_Size": n_train,
            "Algorithm": algo, "Parallel_Model": "None" if seq else C.PARALLEL_MODEL,
            "Hardware": hardware, "Resource_Count": p, "Threads": C.THREADS_PER_WORKER,
            "Execution_Time": t.elapsed, "Communication_Time": comm,
            "Synchronization_Time": sync, "Scheduling_Time": 0.0 if seq else sched,
            "Memory_Use": round(ms.peak_mb, 2), "Run_ID": run_id,
        })
        append_accuracy(acc_csv, {
            "Dataset": dataset, "Problem_Size": n_train, "Workload": workload,
            "Algorithm": algo, "Resource_Count": p, "Run_ID": run_id, "Accuracy": acc})


def run_config(dataset, Xtr, ytr, Q, yq, workers_list, out_dir, hardware):
    n_train, n_query, k = len(Xtr), len(Q), C.K_NEIGHBORS
    raw_csv, acc_csv = out_dir / "raw_results.csv", out_dir / "accuracy.csv"
    workload = f"knn_k{k}_q{n_query}"
    common = dict(dataset=dataset, n_train=n_train, n_query=n_query,
                  hardware=hardware, ytr=ytr, yq=yq, raw_csv=raw_csv, acc_csv=acc_csv)

    def need_for(algo, p):
        have = count_existing(raw_csv, dataset, n_train, workload, algo, p)
        return max(C.N_RUNS - have, 0), have + 1

    need, start = need_for(C.ALG_SEQ, 1)
    if need:
        xsq = np.einsum("ij,ij->i", Xtr, Xtr).astype(np.float32)
        run_series(lambda: (search_sequential(Q, Xtr, xsq, k), 0.0, 0.0), need, start,
                   algo=C.ALG_SEQ, p=1, sched=0.0, **common)

    with SearchContext(Xtr, Q, k) as ctx:
        for p in workers_list:
            nq_, sq_ = need_for(C.ALG_QUERY, p)
            nd_, sd_ = need_for(C.ALG_DATA, p)
            if not (nq_ or nd_):
                continue
            with PoolRunner(ctx, p) as runner:
                if nq_:
                    run_series(runner.query_parallel, nq_, sq_, algo=C.ALG_QUERY, p=p,
                               sched=runner.scheduling_time, **common)
                if nd_:
                    run_series(runner.data_parallel, nd_, sd_, algo=C.ALG_DATA, p=p,
                               sched=runner.scheduling_time, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny offline run on sklearn digits (writes to results/s02_quick)")
    a = ap.parse_args()

    if a.quick:
        name, (X, y) = "digits", load_digits(C.SEED)
        sizes, qcounts, workers = C.QUICK_TRAIN_SIZES, C.QUICK_QUERY_COUNTS, C.QUICK_WORKER_COUNTS
        pool_n, out_dir = C.QUICK_N_QUERY_POOL, C.QUICK_RESULTS_DIR
    else:
        name, (X, y) = "mnist", load_mnist()
        sizes, qcounts, workers = C.TRAIN_SIZES, C.QUERY_COUNTS, C.WORKER_COUNTS
        pool_n, out_dir = C.N_QUERY_POOL, C.RESULTS_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    perf, eff = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
    write_environment_json(out_dir / "environment.json", extra={
        "s02_k_neighbors": C.K_NEIGHBORS, "s02_chunks_per_worker": C.CHUNKS_PER_WORKER,
        "blas_threads_pinned_to": 1, "performance_cores": perf, "efficiency_cores": eff})

    hw = hardware_name()
    Xtr_all, ytr_all, Xq_all, yq_all = split_train_query(X, y, pool_n, C.SEED)
    for n_train in sizes:
        for n_query in qcounts:
            print(f"[{name}] train={n_train} queries={n_query}", flush=True)
            run_config(name, Xtr_all[:n_train], ytr_all[:n_train], Xq_all[:n_query],
                       yq_all[:n_query], workers, out_dir, hw)
    print(f"done -> {out_dir}/raw_results.csv")


if __name__ == "__main__":
    main()
