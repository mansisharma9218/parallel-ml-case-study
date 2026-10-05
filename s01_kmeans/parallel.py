"""Parallel Lloyd's K-means using a persistent multiprocessing.Pool.

Design:
- The dataset is placed once in multiprocessing.shared_memory and never
  re-pickled — each worker opens it by name in a Pool initializer and
  keeps a module-global ndarray view for the Pool's lifetime.
- Each iteration, the (small) current centroids are sent to each worker,
  which assigns its shard of points and returns per-cluster partial sums
  and counts; the main process reduces those into new centroids.
- Per-iteration time is split into:
    * scheduling  — pool creation (once) + task dispatch (enqueueing)
    * communication — IPC/pickling overhead beyond the slowest worker's
      own reported compute time (i.e. time spent waiting on results
      minus actual compute)
    * synchronization — the main process's reduction step itself
  This matches the breakdown named in the experiment spec.
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import time
from multiprocessing import Pool
from multiprocessing import shared_memory

import numpy as np

from .sequential import grouped_sums

_shared_data = None
_shm_handle = None


def _worker_init(shm_name, shape, dtype_name):
    global _shared_data, _shm_handle
    _shm_handle = shared_memory.SharedMemory(name=shm_name)
    _shared_data = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=_shm_handle.buf)


def _worker_partial_fit(start, end, centroids, k):
    t0 = time.perf_counter()
    X_shard = _shared_data[start:end]

    # ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2 via matmul: O(shard*k) memory
    # instead of the O(shard*k*d) a naive broadcasted difference needs.
    centroids32 = centroids.astype(np.float32, copy=False)
    x_sq = np.einsum("ij,ij->i", X_shard, X_shard)[:, None]
    c_sq = np.einsum("ij,ij->i", centroids32, centroids32)[None, :]
    cross = X_shard @ centroids32.T
    dists = x_sq - 2.0 * cross + c_sq
    labels = dists.argmin(axis=1)

    partial_sums, partial_counts = grouped_sums(X_shard, labels, k)

    compute_time = time.perf_counter() - t0
    return partial_sums, partial_counts, compute_time


def _make_shards(n, n_workers):
    bounds = np.linspace(0, n, n_workers + 1).astype(np.int64)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_workers)]


class ParallelKMeansResult:
    def __init__(self, centroids, labels, n_iterations, timing):
        self.centroids = centroids
        self.labels = labels
        self.n_iterations = n_iterations
        self.timing = timing  # dict: scheduling/communication/synchronization/compute (seconds)


def kmeans_parallel(X, k, initial_centroids, n_workers, max_iter=100, tol=1e-4):
    """Same contract as sequential.kmeans_sequential, plus n_workers and
    a returned per-phase timing breakdown (ParallelKMeansResult.timing).
    """
    n, d = X.shape
    X = np.ascontiguousarray(X, dtype=np.float32)

    scheduling_time = 0.0
    communication_time = 0.0
    synchronization_time = 0.0
    compute_time_total = 0.0

    t_pool_create_start = time.perf_counter()
    shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
    try:
        shm_array = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
        shm_array[:] = X[:]

        shards = _make_shards(n, n_workers)

        pool = Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(shm.name, X.shape, X.dtype.name),
        )
        scheduling_time += time.perf_counter() - t_pool_create_start

        try:
            centroids = initial_centroids.astype(np.float64).copy()
            labels = None

            for iteration in range(1, max_iter + 1):
                t_dispatch_start = time.perf_counter()
                async_results = [
                    pool.apply_async(_worker_partial_fit, (start, end, centroids, k))
                    for start, end in shards
                ]
                scheduling_time += time.perf_counter() - t_dispatch_start

                t_gather_start = time.perf_counter()
                results = [r.get() for r in async_results]
                gather_time = time.perf_counter() - t_gather_start

                worker_compute_max = max(r[2] for r in results)
                compute_time_total += worker_compute_max
                communication_time += max(0.0, gather_time - worker_compute_max)

                t_reduce_start = time.perf_counter()
                total_sums = np.zeros((k, d), dtype=np.float64)
                total_counts = np.zeros(k, dtype=np.int64)
                for partial_sums, partial_counts, _ in results:
                    total_sums += partial_sums
                    total_counts += partial_counts

                new_centroids = centroids.copy()
                nonempty = total_counts > 0
                new_centroids[nonempty] = (
                    total_sums[nonempty] / total_counts[nonempty, None]
                )

                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                synchronization_time += time.perf_counter() - t_reduce_start

                if shift < tol:
                    break

            # Final label assignment for the converged centroids (main
            # process; same memory-efficient formula as the workers use).
            centroids32 = centroids.astype(np.float32, copy=False)
            x_sq = np.einsum("ij,ij->i", X, X)[:, None]
            c_sq = np.einsum("ij,ij->i", centroids32, centroids32)[None, :]
            cross = X @ centroids32.T
            labels = (x_sq - 2.0 * cross + c_sq).argmin(axis=1)
        finally:
            pool.close()
            pool.join()
    finally:
        shm.close()
        shm.unlink()

    timing = {
        "scheduling": scheduling_time,
        "communication": communication_time,
        "synchronization": synchronization_time,
        "compute": compute_time_total,
    }
    return ParallelKMeansResult(centroids, labels, iteration, timing)
