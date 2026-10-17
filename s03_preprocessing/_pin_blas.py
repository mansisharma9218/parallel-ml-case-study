"""Pins BLAS to 1 thread. Must be imported before numpy anywhere in this
package, so the sequential baseline is truly single-threaded and parallel
speedup isn't inflated by hidden BLAS-level parallelism. Under the
`spawn` multiprocessing start method each worker process re-imports this
module fresh, so the pin applies there too.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
