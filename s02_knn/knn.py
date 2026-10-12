"""Exact brute-force k-NN (squared Euclidean): a sequential baseline and two
multiprocessing strategies.

  query-parallel : each worker handles a slice of the queries against the full
                   training set (results are just stitched together).
  data-parallel  : each worker holds a slice of the training set and returns its
                   local top-k for ALL queries; the main process merges them.

Timing breakdown (all measured in the main process):
  Communication_Time    = wall time of dispatch->gather minus compute time of the
                          busiest worker (IPC, pickling, dispatch, load imbalance)
  Synchronization_Time  = serial reduction in the main process (merge / stitch)
  Scheduling_Time       = pool creation until every worker is ready (one-off per
                          pool; NOT included in Execution_Time)
"""
from . import _pin_blas  # noqa: F401
import multiprocessing as mp
import os
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np

from common.timing import timer
from . import config as C


# ------------------------------------------------------------------ kernel
def topk_block(Q, X, x_sq, k, offset=0):
    # ||q-x||^2 = ||q||^2 - 2q.x + ||x||^2 ; ||q||^2 is constant per query
    d = x_sq[None, :] - 2.0 * (Q @ X.T)
    k = min(k, X.shape[0])
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    return np.take_along_axis(d, idx, axis=1), idx + offset


def topk_all(Q, X, x_sq, k, offset=0):
    if len(Q) == 0:
        return np.empty((0, k), np.float32), np.empty((0, k), np.int64)
    D, I = [], []
    for s in range(0, len(Q), C.QUERY_BLOCK):
        d, i = topk_block(Q[s:s + C.QUERY_BLOCK], X, x_sq, k, offset)
        D.append(d)
        I.append(i)
    return np.vstack(D), np.vstack(I)


def vote(labels_nn, n_classes=10):
    counts = np.zeros((labels_nn.shape[0], n_classes), dtype=np.int32)
    rows = np.arange(labels_nn.shape[0])
    for j in range(labels_nn.shape[1]):
        np.add.at(counts, (rows, labels_nn[:, j]), 1)
    return counts.argmax(1)


def search_sequential(Q, X, x_sq, k):
    return topk_all(Q, X, x_sq, k)[1]


# ------------------------------------------------------------------ workers
_G = {}


def _attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # py>=3.13
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _init_worker(names, shapes, dtypes, k, barrier):
    shms = []
    for key in names:
        shm = _attach(names[key])
        shms.append(shm)
        _G[key] = np.ndarray(shapes[key], dtype=dtypes[key], buffer=shm.buf)
    _G["shms"], _G["k"] = shms, k
    barrier.wait(timeout=120)  # parent waits too -> measures real readiness


def _query_task(span):
    t0 = time.perf_counter()
    s, e = span
    _, idx = topk_all(_G["Q"][s:e], _G["X"], _G["xsq"], _G["k"])
    return os.getpid(), s, idx, time.perf_counter() - t0


def _data_task(span):
    t0 = time.perf_counter()
    s, e = span
    d, idx = topk_all(_G["Q"], _G["X"][s:e], _G["xsq"][s:e], _G["k"], offset=s)
    return os.getpid(), d, idx, time.perf_counter() - t0


# ------------------------------------------------------------------ parallel
class SearchContext:
    """Owns the shared-memory copies of training data, norms and queries."""

    def __init__(self, X, Q, k):
        self.k, self.n_train, self.n_query = k, len(X), len(Q)
        self.xsq = np.einsum("ij,ij->i", X, X).astype(np.float32)
        arrays = {"X": X, "Q": Q, "xsq": self.xsq}
        self._shms, self.names, self.shapes, self.dtypes = [], {}, {}, {}
        for key, arr in arrays.items():
            shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
            np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)[:] = arr
            self._shms.append(shm)
            self.names[key], self.shapes[key], self.dtypes[key] = shm.name, arr.shape, arr.dtype

    def close(self):
        for s in self._shms:
            s.close()
            try:
                s.unlink()
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class PoolRunner:
    """A persistent pool of `workers` processes attached to a SearchContext."""

    def __init__(self, ctx, workers):
        self.ctx, self.workers = ctx, workers
        mpctx = mp.get_context("spawn")
        barrier = mpctx.Barrier(workers + 1)
        with timer() as t:
            self.pool = mpctx.Pool(
                workers, initializer=_init_worker,
                initargs=(ctx.names, ctx.shapes, ctx.dtypes, ctx.k, barrier))
            barrier.wait(timeout=120)
        self.scheduling_time = t.elapsed

    @staticmethod
    def _comm(wall, pid_times):
        per_pid = defaultdict(float)
        for pid, el in pid_times:
            per_pid[pid] += el
        return max(wall - max(per_pid.values()), 0.0)

    def query_parallel(self):
        nq, k = self.ctx.n_query, self.ctx.k
        edges = np.linspace(0, nq, self.workers * C.CHUNKS_PER_WORKER + 1, dtype=int)
        tasks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        t0 = time.perf_counter()
        res = self.pool.map(_query_task, tasks, chunksize=1)
        wall = time.perf_counter() - t0
        comm = self._comm(wall, [(r[0], r[3]) for r in res])
        with timer() as ts:
            out = np.empty((nq, k), dtype=np.int64)
            for _, s, idx, _ in res:
                out[s:s + len(idx)] = idx
        return out, comm, ts.elapsed

    def data_parallel(self):
        n, k = self.ctx.n_train, self.ctx.k
        edges = np.linspace(0, n, self.workers + 1, dtype=int)
        tasks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        t0 = time.perf_counter()
        res = self.pool.map(_data_task, tasks, chunksize=1)
        wall = time.perf_counter() - t0
        comm = self._comm(wall, [(r[0], r[3]) for r in res])
        with timer() as ts:
            D = np.hstack([r[1] for r in res])
            I = np.hstack([r[2] for r in res])
            sel = np.argpartition(D, k - 1, axis=1)[:, :k]
            out = np.take_along_axis(I, sel, axis=1)
        return out, comm, ts.elapsed

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
