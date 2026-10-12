"""Pin every BLAS/OpenMP backend to 1 thread so the sequential baseline is
genuinely single-threaded and all parallelism comes from our worker processes.
VECLIB_MAXIMUM_THREADS covers Apple Accelerate (NumPy's default on Apple Silicon).
"""
import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"
