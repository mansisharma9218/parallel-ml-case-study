# S02 — Parallel k-NN Search

Exact brute-force k-NN (k=5, squared Euclidean) on MNIST: a sequential baseline vs. two
`multiprocessing.Pool` strategies, with execution time split into communication /
synchronization / scheduling overhead. Accuracy is recorded separately from runtime.

## Setup
Same environment as S01 (`pip install -r requirements.txt` from the repo root). No new dependencies.
Before a full run, set `STUDENT_ID` in `s02_knn/config.py`.

## Modules
| Module | Job |
|---|---|
| `data_loading.py` | MNIST with labels (cached to `data/raw/mnist_s02.npz`); last 10k images = query pool, rest = training data |
| `knn.py` | Sequential baseline; query-parallel and data-parallel strategies; persistent pool, data in `shared_memory` |
| `experiment_runner.py` | Runs the train-size x query-count x workers grid, 1 warm-up + 5 timed runs each; resume-safe |
| `analysis.py` | Speedup / Efficiency / queries-per-sec from raw times; writes `summary.csv` and plots |
| `memory_sampler.py` | Peak USS memory (main + workers) sampled during each timed run |
| `_pin_blas.py` | Pins BLAS (incl. Apple Accelerate) to 1 thread |
| `config.py` | Fixed constants (sizes, query counts, worker counts, k, seed) |

## Reproducing (run from the repo root)
```
python3 -m s02_knn.experiment_runner --quick     # offline sanity run (sklearn digits)
python3 -m s02_knn.analysis --quick
python3 -m s02_knn.experiment_runner             # full MNIST run (reference machine only)
python3 -m s02_knn.analysis
python3 -m pytest s02_knn/tests/ -v
```
Writes `results/s02/{raw_results.csv, accuracy.csv, summary.csv, environment.json, plots/}`.

## Notes on the data
- `Workload` encodes `knn_k<K>_q<num_queries>`; `Problem_Size` is the training-set size.
- `Execution_Time` is search time with a persistent pool; `Scheduling_Time` (pool start-up until all
  workers are ready) is a one-off per pool and is NOT included in `Execution_Time`.
- `Communication_Time` = dispatch→gather wall time minus the busiest worker's compute time
  (IPC + pickling + load imbalance). `Synchronization_Time` = serial merge in the main process.
- `Speedup`/`Efficiency` are left empty in `raw_results.csv` and computed only in `analysis.py`.
- Accuracy lives in `accuracy.csv` (not in the shared schema) and is computed outside the timed region.
- `Memory_Use` is peak USS, so the shared-memory dataset is largely invisible to it (same caveat as S01).
