# S03 — Parallel Data Preprocessing

CS10 · BCSE412L parallel ML case study.

Research question: *which preprocessing stages scale well under parallel
execution, and which remain bottlenecks?* A four-stage MNIST pipeline is
run sequentially and with two `multiprocessing` strategies (stage-level and
pipeline), with per-stage times, overhead breakdown and CPU utilization
recorded for every run.

## Setup

Same environment as S01/S02 (`pip install -r requirements.txt` from the repo
root). No new dependencies. Before a full run, set `STUDENT_ID` in
`s03_preprocessing/config.py` (the runner refuses a full run until you do).

## The pipeline

| Stage | What it does | Expected behaviour |
|---|---|---|
| `load` | reads raw uint8 rows from a memory-mapped `.npy`, casts to float32 | I/O / memory-bandwidth bound |
| `normalize` | per-feature min-max scaling to [0, 1] | memory-bound, and needs a **global** reduction (column min/max) before any row can be scaled |
| `deskew` | moment-based shear correction of each image (`scipy.ndimage.affine_transform`) | compute-heavy, fully independent per image |
| `pca` | per-chunk Gram matrices → reduction → **serial** eigendecomposition → per-row projection to 50 components | parallel Gram/projection, but a serial core (Amdahl) |

## Strategies compared

| Algorithm | How work is split |
|---|---|
| `Preproc-Sequential` | baseline: all stages in the main process, BLAS pinned to 1 thread |
| `Preproc-Stage-Parallel` | **stage-level parallelism**: each stage is split into chunks over all workers, with a barrier after every step (6 `pool.map` phases) |
| `Preproc-Pipeline-Parallel` | **pipeline (chunk-streaming)**: each task carries one chunk through consecutive stages until a global statistic is needed, so workers can be in different stages at once and chunks stay cache-warm; only the 2 unavoidable barriers remain (3 phases) |

Both parallel strategies use a `spawn` pool whose workers read their own input
rows from disk and work on `shared_memory` buffers (nothing large is pickled).
Like S01, the pool is created inside each timed run, so `Execution_Time`
includes pool start-up (S02 instead reuses one pool and excludes it — keep
that in mind when comparing across sub-problems).

## Module map

| Module | Job |
|---|---|
| `stages.py` | the stage kernels, shared by all three strategies so they do identical arithmetic |
| `sequential.py` | baseline + the `PreprocResult` container |
| `parallel.py` | stage-level and pipeline strategies; overhead accounting |
| `data_loading.py` | downloads MNIST once into `data/raw/` (shared OpenML cache with S01) as uncompressed `.npy` |
| `experiment_runner.py` | full grid: sizes × strategies × workers, 1 warm-up + 5 timed runs; resume-safe |
| `analysis.py` | speedup / efficiency / throughput from raw times; summaries + plots |
| `preprocess.py` | writes the processed files S01/S02 consume (below) |
| `resource_sampler.py` | per-run peak memory (USS, main + workers) and system-wide CPU utilization |
| `config.py` | IDs and fixed experiment constants |
| `_pin_blas.py` | pins BLAS to 1 thread (same as S01) |

## Reproducing (run from the repo root)

```bash
python -m pytest s03_preprocessing/tests/ -v        # correctness (offline)

python -m s03_preprocessing.experiment_runner --quick   # sanity run -> results/s03_quick (git-ignored)
python -m s03_preprocessing.analysis --quick

python -m s03_preprocessing.experiment_runner       # full grid (reference machine only)
python -m s03_preprocessing.analysis

python -m s03_preprocessing.preprocess              # writes data/processed/* for S01/S02
```

The full grid is 3 sizes (17.5k / 35k / 70k images) × (sequential + 2
strategies × 1/2/4/8 workers) × 6 runs; expect roughly 15–25 minutes.

On a machine that throttles under sustained load (see *Threats to
validity*), add `--cooldown <seconds>` to pause, untimed, before every run;
the value is recorded in `environment.json`. Because the runner is
resume-safe, a size can be re-measured by deleting its rows from
`raw_results.csv`, `stages.csv` and `run_metrics.csv` and re-running.

### Outputs

`results/s03/`:
- `raw_results.csv` — one shared-schema row per timed run (`Workload` = `preproc_pca50`, `Problem_Size` = number of images)
- `stages.csv` — per run and stage: `Wall_Time` (empty for the pipeline strategy, whose stages are fused) and `Busy_Time` (summed over all processes)
- `run_metrics.csv` — per run: `Data_Movement_Time`, `CPU_Utilization_Pct` (not in the shared schema)
- `summary.csv` — median/std time, Speedup, Efficiency, throughput (images/s), median overheads, CPU utilization
- `stage_summary.csv` — per-stage median wall/busy time, stage speedup, share of stage time
- `environment.json`
- `plots/` (each as `.png` and editable `.svg`): `stage_contribution_*`, `speedup_vs_workers_*`, `stage_speedup_vs_workers_*`, `throughput_vs_workers_*`, `time_breakdown_*`, `cpu_utilization_vs_workers_*`

`data/processed/` (git-ignored, from `preprocess.py`):
- `mnist_normalized.npy` — float32 (70000, 784) in [0, 1]: the agreed S01 input. Bit-identical to S01's own fallback normalization (checked), so S01's results don't change when it switches to this file.
- `mnist_labels.npy` — int64 (70000,), same row order (for S02, which needs labels)
- `mnist_deskewed_pca50.npy` — float32 (70000, 50): the full-pipeline features

## Notes on the measurements

- `Execution_Time` — wall time of the whole pipeline call, including pool
  start-up, shared-memory set-up and copying results out.
- `Scheduling_Time` — pool creation until all workers pass a readiness
  Barrier, plus pool shutdown.
- `Communication_Time` — per `pool.map`: wall time minus the busiest
  worker's busy time (dispatch, pickling, result return *and load
  imbalance*), summed over phases.
- `Synchronization_Time` — main-process reductions of partial results
  (column min/max; per-worker Gram slots). The serial eigendecomposition is
  counted as compute (stage `pca`), not synchronization.
- `Data_Movement_Time` (`run_metrics.csv`) — shared-memory allocation/release
  and copying the outputs back out.
- `Speedup`/`Efficiency` are left empty in `raw_results.csv` and computed
  only in `analysis.py`, against the sequential baseline for the same size.
- `Memory_Use` is peak USS of main + workers; the shared-memory buffers are
  largely invisible to USS (same caveat as S01).
- `CPU_Utilization_Pct` is system-wide, so other load on the machine shows up
  in it.
- The resource sampler polls every 0.2 s and starts/stops outside the timed
  region; at S01's 0.05 s it measurably slowed the Python-loop deskew stage
  in the main process.

## Threats to validity

- **Machine state distorts the baseline.** On a Windows laptop (12th-gen
  Intel i5-1240P, 16 GB, performance + efficiency cores) two effects were
  seen: (1) a long-running process launched from a background terminal
  roughly halved its single-core speed after ~25 s of load, which made
  the first n=70,000 measurements cost ~2.5x more per image than at
  17.5k/35k, even with a 30 s untimed cool-down before each run;
  (2) launched as a separate process, one 70k run still showed the
  1-worker runs slowing from 5.9 s to ~11 s over consecutive runs, while
  a standalone check of the same configuration gave a steady 6.0 s --
  a transient background disturbance (the repo sits in a OneDrive-synced
  folder). Both effects produce impossible or inflated results (e.g.
  speedup > 1 with one worker). The committed n=70,000 rows come from a
  clean re-run launched as a separate process. Sanity check before
  trusting a size: per-image stage cost must be roughly constant across
  sizes (it is: deskew ~48-51 us, PCA ~20-24 us per image at all three
  sizes), 1-worker speedup must be < 1, and the 5 runs of a configuration
  should agree closely. Run on mains power, with other applications
  closed and file sync paused, and report medians.
- The `load` stage reads from the OS page cache after the first run, so it
  measures memory bandwidth more than disk I/O.
- The pipeline strategy is chunk-streaming (each worker runs consecutive
  stages on its chunk), not a one-stage-per-worker assembly line; with two
  global-statistic barriers in this pipeline, a dedicated-stage design would
  stall at each barrier anyway.
