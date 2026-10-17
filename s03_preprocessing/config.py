"""Identifiers and fixed experiment constants shared across s03_preprocessing
modules, so every raw CSV row and environment record is consistent.
"""

STUDENT_ID = "23BCE0854"
BATCH_ID = "CS10"

DATASET = "mnist"
PROBLEM_SIZES = [17_500, 35_000, 70_000]  # first n MNIST images
WORKER_COUNTS = [1, 2, 4, 8]

N_COMPONENTS = 50  # PCA output dimensionality
CHUNKS_PER_WORKER = 4  # tasks per worker per parallel phase (load balancing)

N_WARMUP_RUNS = 1
N_TIMED_RUNS = 5

# --quick: tiny sanity run, written to results/s03_quick (git-ignored)
QUICK_PROBLEM_SIZES = [2_000, 5_000]
QUICK_WORKER_COUNTS = [1, 2]
QUICK_N_TIMED_RUNS = 1

ALG_SEQ = "Preproc-Sequential"
ALG_STAGE = "Preproc-Stage-Parallel"
ALG_PIPE = "Preproc-Pipeline-Parallel"
PARALLEL_MODEL = "multiprocessing.Pool+shared_memory"
THREADS_PER_WORKER = 1

WORKLOAD = f"preproc_pca{N_COMPONENTS}"
