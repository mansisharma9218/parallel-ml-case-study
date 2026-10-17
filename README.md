# parallel-ml-case-study

CS10 · BCSE412L parallel ML case study — **Parallel Machine Learning**.

How well do common ML workloads scale when parallelised across CPU cores
with Python `multiprocessing`? Each sub-problem compares a single-threaded
sequential baseline (BLAS pinned to 1 thread) against one or more parallel
strategies at 1/2/4/8 workers, and splits execution time into compute,
communication, synchronization and scheduling overhead. All three write
their raw results in one shared CSV schema so they can be merged.

| Sub-problem | Package | Student ID | Question |
|---|---|---|---|
| S01 — Parallel K-means | [`s01_kmeans/`](s01_kmeans/README.md) | 23BCE0856 | How does Lloyd's K-means scale with data size, K and workers? |
| S02 — Parallel k-NN search | [`s02_knn/`](s02_knn/README.md) | 23BCE0905 | Query-parallel vs. data-parallel exact k-NN: which scales better? |
| S03 — Parallel data preprocessing | [`s03_preprocessing/`](s03_preprocessing/README.md) | 23BCE0854 | Which preprocessing stages scale, and which remain bottlenecks? |

Each package's README has the full design, module map and measurement
notes; this file is the overview.

## Repository layout

```
common/              shared harness: CSV schema, CSV writer, timer, environment record
                     (changes only via a PR all three teammates approve)
s01_kmeans/          S01 code + tests
s02_knn/             S02 code + tests
s03_preprocessing/   S03 code + tests
results/s01|s02|s03/ committed results: raw_results.csv, summary.csv,
                     environment.json, plots/ (plus sub-problem extras)
data/                downloaded / processed datasets (git-ignored, recreated on first run)
requirements.txt     one environment for all three sub-problems
```

## Setup

Python 3.11+ (results were produced on 3.12–3.14). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MNIST is downloaded from OpenML on first use (network needed once) and
cached under `data/`. No GPU is used.

## Running

All commands run from the repo root (the packages use relative imports).
On macOS/Linux use `python3` if `python` isn't Python 3.

**Tests** (offline, about 15 s):

```bash
python -m pytest s01_kmeans/tests s02_knn/tests s03_preprocessing/tests -q
```

**Quick sanity runs** (tiny sizes, under a minute each):

```bash
python -m s02_knn.experiment_runner --quick            # -> results/s02_quick (git-ignored)
python -m s03_preprocessing.experiment_runner --quick  # -> results/s03_quick (git-ignored)
```

> Note: `s01_kmeans.experiment_runner --quick` currently writes into the
> committed `results/s01/` folder (appending test rows and overwriting
> `environment.json`), so don't run it inside the repo until that is fixed.

**Full experiments, then analysis** (each runner is resume-safe: re-running
skips configurations that already have all their timed runs):

```bash
python -m s01_kmeans.experiment_runner        && python -m s01_kmeans.analysis
python -m s02_knn.experiment_runner           && python -m s02_knn.analysis
python -m s03_preprocessing.experiment_runner && python -m s03_preprocessing.analysis
python -m s03_preprocessing.preprocess        # writes data/processed/* (the shared MNIST files)
```

Run full experiments on mains power with other applications closed;
laptop throttling visibly distorted some measurements (see the per-package
READMEs).

## Shared results format

Every `results/<sub-problem>/raw_results.csv` has exactly the columns in
`common/csv_schema.py`, one row per timed run (1 warm-up + 5 timed runs per
configuration). `Speedup` and `Efficiency` are left empty there and
computed only by each `analysis.py` (against the sequential baseline for
the same dataset/size/workload), which writes `summary.csv` and `plots/`.
`environment.json` records the machine and software versions. Extra,
sub-problem-specific files (`iterations.csv`, `accuracy.csv`, `stages.csv`,
`run_metrics.csv`, ...) are described in each package README.

## Results at a glance

| | S01 K-means | S02 k-NN | S03 Preprocessing |
|---|---|---|---|
| Machine | Apple M2, 8 GB, Python 3.12.6 | Apple M5, 16 GB, Python 3.14.7 | Intel i5-1240P, 16 GB, Windows 11, Python 3.13.7 |
| Data | synthetic blobs (100k / 500k / 2M × 32) + MNIST 70k; K = 8 / 16 / 64 | MNIST, 6k–60k training images, 1k / 5k / 10k queries, k = 5 | MNIST, 17.5k / 35k / 70k images |
| Parallel strategies | data-parallel Lloyd's (pool + shared memory) | query-parallel; data-parallel | stage-level; pipeline (chunk-streaming) |
| Best speedup | about 3.5× (K=8, 2M points, 8 workers) | about 5.0× (query-parallel, 10k queries, 8 workers) | about 1.8× (stage-level, 70k, 4 workers) |
| Correctness check | same iteration counts as sequential in every configuration | identical accuracy (91–97%) for every strategy and worker count | normalized output bit-identical to S01's own normalization |

Main findings:

- **More work per data item scales better.** k-NN (every query against
  every training point) scales best; K-means reaches about 3–3.5× at large
  sizes; preprocessing is mostly cheap, memory-bound steps and tops out
  at about 1.8×.
- **Fixed parallel overhead dominates small problems.** Pool start-up
  costs roughly 0.7–1.5 s, so the smallest S03 size never beats sequential,
  and 8 workers is often slower than 4 in S01 and S03.
- **Global synchronization points cap speedup (Amdahl).** In S03,
  normalization needs global column min/max and PCA needs the full
  covariance plus a serial eigendecomposition; deskew scales well, while
  PCA and pool start-up become the bottleneck.

## Comparing across sub-problems — read before combining

- **Different machines.** Each sub-problem ran on different hardware and
  Python/NumPy versions (table above). Compare speedup and efficiency
  *trends*, not absolute times, and label the machine in any combined table.
- **`Scheduling_Time` differs.** S01 and S03 create the pool inside each
  timed run, so pool start-up is included in `Execution_Time`. S02 reuses
  one pool and reports its start-up only in `Scheduling_Time`, excluded
  from `Execution_Time`, which makes S02's speedups somewhat more favourable.
- **`Communication_Time`** is, in all three, the wall time of each parallel
  step minus the busiest worker's compute time: inter-process transfer
  *plus load imbalance*. It is small because the large arrays live in
  shared memory and are never pickled.
- **`Memory_Use`** is peak USS (unique pages) of the main process and
  workers, so shared-memory data is largely not counted.
- **MNIST scaling.** S01 and S03 scale each pixel column by its own
  min/max (identical results); S02 divides every pixel by 255.
- **S01 K-means:** most runs stop at the 100-iteration cap rather than
  converging (tolerance 1e-4), so their times are "time for 100
  iterations"; K=8 at 500k converges in 4 iterations, which is why it is
  faster than 100k.
