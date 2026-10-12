"""Fixed experiment constants for S02 (parallel k-NN)."""
from pathlib import Path

# TODO: set your own registration number before the full run.
STUDENT_ID = "23BCE0905"
BATCH_ID = "CS10"

K_NEIGHBORS = 5
SEED = 42
N_WARMUP = 1
N_RUNS = 5

WORKER_COUNTS = [1, 2, 4, 8]
TRAIN_SIZES = [6000, 15000, 30000, 60000]
QUERY_COUNTS = [1000, 5000, 10000]
N_QUERY_POOL = 10000  # last 10k MNIST images = official test split

# --quick (sklearn digits, offline)
QUICK_TRAIN_SIZES = [300, 800, 1400]
QUICK_QUERY_COUNTS = [100, 300]
QUICK_WORKER_COUNTS = [1, 2, 4]
QUICK_N_QUERY_POOL = 360

QUERY_BLOCK = 256        # queries per distance block (bounds memory)
CHUNKS_PER_WORKER = 4    # query-parallel: tasks per worker (load balancing)

ALG_SEQ = "kNN-Sequential"
ALG_QUERY = "kNN-Parallel-Query"
ALG_DATA = "kNN-Parallel-Data"
PARALLEL_MODEL = "multiprocessing.Pool"
THREADS_PER_WORKER = 1

RESULTS_DIR = Path("results/s02")
QUICK_RESULTS_DIR = Path("results/s02_quick")
