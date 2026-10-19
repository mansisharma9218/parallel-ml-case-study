# S01 — Parallel K-means

Mansi Sharma (23BCE0856) · Batch CS10 · BCSE412L parallel ML case study.

Sequential vs. multiprocessing-parallel Lloyd's K-means, measured on
synthetic Gaussian blobs and MNIST, with execution time split into
compute / communication (IPC) / synchronization (reduction) / scheduling
(pool creation + dispatch) overhead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate   # from the repo root
pip install -r requirements.txt
```

Python 3.11+ (developed/tested on 3.12). No GPU used.

## What's in this package

| Module | Job |
|---|---|
| `data_loading.py` | Generates synthetic Gaussian blobs; loads MNIST from S03's agreed output (`data/processed/mnist_normalized.npy`), falling back to downloading + normalizing it if that isn't there yet |
| `sequential.py` | Baseline: vectorized NumPy Lloyd's K-means, single-threaded |
| `parallel.py` | Parallel Lloyd's K-means: persistent `multiprocessing.Pool`, dataset in `shared_memory`, workers return partial sums/counts, main process reduces |
| `experiment_runner.py` | Runs the full dataset x size x K x workers grid, 1 warm-up + 5 timed runs each |
| `analysis.py` | Computes Speedup/Efficiency from raw times, writes `summary.csv` and the plots |
| `config.py` | Student/Batch IDs and the fixed experiment constants (sizes, K values, worker counts, seed) |
| `_pin_blas.py` | Pins BLAS to 1 thread; imported first by every module that uses NumPy so the sequential baseline is genuinely single-threaded |

## Reproducing the results

All commands below are run from the repo root (not inside `s01_kmeans/`),
since the package uses relative imports.

**1. Quick sanity run** (tiny sizes, finishes in well under a minute —
do this before a full run to catch problems early):

```bash
python3 -m s01_kmeans.experiment_runner --quick
```

`--quick` writes to `results/s01_quick/` (git-ignored), not `results/s01/`;
run `python3 -m s01_kmeans.analysis --quick` to analyse it.

Add `--no-mnist` to skip MNIST (useful offline — MNIST download needs
network access the first time it's used).

**2. Full run** (only on the agreed reference machine, so results are
comparable across the team):

```bash
python3 -m s01_kmeans.experiment_runner
```

This sweeps synthetic blobs at 100k / 500k / 2M points, MNIST (70k x
784), K in {8, 16, 64}, and 1/2/4/8 workers — a sequential baseline plus
4 parallel configurations per (dataset, size, K) combination, 5 timed
runs each. It's compute-heavy (especially the 2M-point, K=64 sequential
runs); expect it to take a while. Writes:

- `results/s01/raw_results.csv` — one row per timed run (shared schema;
  see note below on how K is encoded)
- `results/s01/iterations.csv` — iterations-to-convergence per run (K-means-specific, not part of the shared schema)
- `results/s01/environment.json` — CPU/RAM/OS/Python/NumPy snapshot, recorded once

**3. Analysis + plots:**

```bash
python3 -m s01_kmeans.analysis
```

Reads `raw_results.csv`, computes Speedup (vs. the sequential baseline
for the same dataset/size/K) and Efficiency (`Speedup / Resource_Count`)
from the raw times, and writes:

- `results/s01/summary.csv` — median/std execution time, speedup, efficiency per configuration
- `results/s01/plots/time_vs_samples_k<K>.png`
- `results/s01/plots/time_per_iteration_vs_samples_k<K>.png`
- `results/s01/plots/speedup_vs_workers_<dataset>_n<size>.png`
- `results/s01/plots/time_breakdown_<dataset>_n<size>_k<K>.png` — stacked compute/IPC/reduction/scheduling
- `results/s01/plots/iterations_vs_configuration.png`

Each plot is saved as both `.png` and `.svg` (editable).

## Tests

```bash
python3 -m pytest s01_kmeans/tests/ -v
```

- `test_sequential_vs_sklearn.py` — the sequential baseline matches
  `sklearn.cluster.KMeans` (same initial centroids, `n_init=1`)
- `test_parallel_vs_sequential.py` — the parallel version matches the
  sequential baseline for 1/2/4 workers, and reports a non-negative
  timing breakdown
- `test_grouped_sums.py` — the vectorized per-cluster reduction
  (`grouped_sums`, below) matches a naive per-cluster loop exactly

## Notes on the data

- `Workload` encodes K as `kmeans_k<K>` (e.g. `kmeans_k64`), since the
  shared raw-results schema (`common/csv_schema.py`) doesn't have a
  dedicated K column — `analysis.py` parses it back out.
- `Speedup`/`Efficiency` are left empty in `raw_results.csv` and computed
  only in `analysis.py`, from the raw times — never typed in by hand.
- The distance computation uses `||x-c||^2 = ||x||^2 - 2x.c + ||c||^2`
  via matmul rather than a naive broadcasted difference, which keeps
  memory at O(n*k) instead of O(n*k*d) — relevant at n=2M, K=64, d=32,
  where the naive form would need ~33GB.
- The per-cluster centroid-sum reduction (`sequential.grouped_sums`) uses
  one `np.bincount(weights=...)` per feature column on a transposed copy
  of the data instead of `np.add.at`, which is much slower at this scale.
- `multiprocessing.Pool(...)` returns once worker *processes* exist, not
  once they've actually finished starting up (importing numpy, opening
  shared memory). Left alone, that startup cost would land inside the
  first iteration's result-gathering wait and get misattributed as
  communication overhead, so `parallel.py` uses a `multiprocessing.Barrier`
  to make `Scheduling_Time` wait for workers to actually be ready.
- The parallel run skips the final full-dataset label pass after
  converging — the sequential baseline gets labels for free from its
  last iteration, but redoing that serially in the parallel path would
  unfairly inflate its time, and nothing downstream uses the labels
  anyway. `ParallelKMeansResult.labels` is always `None`.
- `Memory_Use` is peak USS (unique, non-shared pages) summed across the
  main process and any live worker children, sampled by a background
  thread during each individual timed run (`memory_sampler.py`) — not a
  whole-process-lifetime peak, and it does see worker memory. Caveat:
  because it's USS, the large `shared_memory`-backed dataset (mapped
  into every worker) is largely invisible to this measurement, since no
  single process uniquely owns those pages — a known limitation, not a
  bug, worth stating in the report rather than treating this column as
  an exact figure.
- `experiment_runner.py` is resume-safe: before each configuration it
  counts existing rows in `raw_results.csv` for that exact
  (Dataset, Problem_Size, Workload, Algorithm, Resource_Count) and skips
  or tops up rather than duplicating, so an interrupted run can just be
  re-invoked.
- `Hardware` uses the actual chip name on macOS (`sysctl -n
  machdep.cpu.brand_string`, e.g. "Apple M2") rather than the generic
  `platform.machine()` string ("arm64"), falling back to the latter on
  other platforms.

## Caveats for the report

- **Iteration counts.** 225 of 300 runs stop at the 100-iteration cap
  rather than converging (tolerance 1e-4 on absolute centroid movement is
  strict), so for most settings "time" means "time for 100 iterations".
  At k=8 the 500k dataset converges in 4 iterations while 100k takes 80,
  which makes `time_vs_samples_k8` look non-monotonic. Use the
  `time_per_iteration_vs_samples_k*.png` plots for scaling: roughly
  13 -> 88 -> 269 ms per iteration for 100k -> 500k -> 2M.
- **k=64, 2M points was re-run.** The first pass was distorted by
  throttling (8-worker runs slowed 16.3 -> 32.5 s across repetitions, one
  1-worker run took 74.4 s), which made 8 workers look slower than 4. It was
  re-run on AC power with the laptop kept awake (the 2-worker setting was
  re-run once more after the laptop slept mid-run). Only these rows changed;
  8 workers now takes 14.8 s median (3.28x) versus 17.2 s for 4 (2.82x).
  Note that the 4-worker runs still creep up slightly (15.8 -> 17.7 s).
  The original first-pass rows are kept, unaltered, in
  `results/s01/throttled_first_pass_k64_2M.csv` as a documented outlier and
  are not used in `summary.csv` or the plots.
- **MNIST source.** S01's MNIST came from its own copy
  (`mnist_normalized_fallback.npy`) because S03's file did not exist yet;
  S03's `data/processed/mnist_normalized.npy` was checked to be
  bit-identical, so results remain valid.
- **One worker slightly faster than sequential** (up to ~6%, e.g. MNIST
  k=8, 1.06x), most likely a cache effect from chunking.
- **`Memory_Use` under-counts shared data** (USS excludes the shared
  dataset, see above).
- **Hardware.** Apple M2, 8 GB, Python 3.12.6; other sub-problems ran on
  different machines, so combined tables need a hardware label.
