"""Two multiprocessing strategies for the preprocessing pipeline. Both use
a pool of worker processes attached to shared-memory buffers; workers
read their own input rows straight from the memory-mapped raw file, and
intermediate arrays are never pickled.

  stage-level : every stage is split into chunks across all workers, with
                a full barrier (one pool.map) after each step -- 6 maps:
                load | min/max stats | normalize | deskew | Gram | project.
  pipeline    : each task carries one chunk through consecutive stages
                until the next point where a global statistic is needed,
                so different workers can be in different stages at the
                same time and a chunk stays cache-warm between stages.
                Only the two unavoidable barriers remain (column min/max
                before scaling; covariance before projecting) -- 3 maps:
                load+stats | normalize+deskew+Gram | project.

Timing breakdown (all measured in the main process):
  scheduling      pool creation until every worker is ready (a Barrier,
                  as in S01) plus pool shutdown
  communication   per pool.map: wall time minus the busiest worker's busy
                  time (dispatch, pickling, result return and load
                  imbalance), summed over all maps
  synchronization main-process reductions of partial results (column
                  min/max; per-worker Gram slots)
  data_movement   shared-memory allocation/release and copying the final
                  outputs back out of shared memory
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import multiprocessing as mp
import os
import time
from collections import defaultdict
from multiprocessing import shared_memory

import numpy as np

from common import timer

from . import stages as S
from .sequential import PreprocResult, raw_geometry

STRATEGIES = ("stage", "pipeline")

_G = {}


def _attach(name):
    try:
        return shared_memory.SharedMemory(name=name, track=False)  # py>=3.13
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _init_worker(raw_path, specs, side, counter, barrier):
    shms = []
    for key, (name, shape, dtype) in specs.items():
        shm = _attach(name)
        shms.append(shm)
        _G[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    with counter.get_lock():
        _G["wid"] = counter.value  # this worker's private Gram slot
        counter.value += 1
    _G["shms"], _G["raw_path"], _G["side"] = shms, raw_path, side
    barrier.wait(timeout=120)  # parent waits too -> Scheduling_Time sees real readiness


def _timed(busy, stage, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    busy[stage] = busy.get(stage, 0.0) + time.perf_counter() - t0
    return out


def _accumulate_gram(D):
    gram, colsum = S.gram_partial(D)
    _G["gram"][_G["wid"]] += gram
    _G["colsum"][_G["wid"]] += colsum


# ---- tasks: each takes a (start, stop) row span, returns (pid, busy, payload)

def _task_load(span):
    s, e = span
    busy = {}
    _timed(busy, "load", S.load_into, _G["raw_path"], s, e, _G["X"])
    return os.getpid(), busy, None


def _task_minmax(span):
    s, e = span
    busy = {}
    stats = _timed(busy, "normalize", S.minmax_partial, _G["X"][s:e])
    return os.getpid(), busy, stats


def _task_normalize(span):
    s, e = span
    busy = {}
    _timed(busy, "normalize", S.normalize_inplace, _G["X"][s:e], _G["col_min"], _G["span"])
    return os.getpid(), busy, None


def _task_deskew(span):
    s, e = span
    busy = {}
    _timed(busy, "deskew", S.deskew_into, _G["X"][s:e], _G["D"][s:e], _G["side"])
    return os.getpid(), busy, None


def _task_gram(span):
    s, e = span
    busy = {}
    _timed(busy, "pca", _accumulate_gram, _G["D"][s:e])
    return os.getpid(), busy, None


def _task_project(span):
    s, e = span
    busy = {}
    _timed(busy, "pca", S.project_into, _G["D"][s:e], _G["mean"], _G["V"], _G["Z"][s:e])
    return os.getpid(), busy, None


def _task_load_stats(span):
    s, e = span
    busy = {}
    _timed(busy, "load", S.load_into, _G["raw_path"], s, e, _G["X"])
    stats = _timed(busy, "normalize", S.minmax_partial, _G["X"][s:e])
    return os.getpid(), busy, stats


def _task_normalize_deskew_gram(span):
    s, e = span
    busy = {}
    _timed(busy, "normalize", S.normalize_inplace, _G["X"][s:e], _G["col_min"], _G["span"])
    _timed(busy, "deskew", S.deskew_into, _G["X"][s:e], _G["D"][s:e], _G["side"])
    _timed(busy, "pca", _accumulate_gram, _G["D"][s:e])
    return os.getpid(), busy, None


class _ParallelRun:
    """Owns the shared-memory buffers and the pool for one pipeline run."""

    def __init__(self, raw_path, n, n_components, workers, chunks_per_worker):
        n_avail, d, side, k = raw_geometry(raw_path, n_components)
        if n > n_avail:
            raise ValueError(f"Asked for {n} rows but {raw_path} has {n_avail}")
        self.n, self.k = n, k
        self.timing = {"communication": 0.0, "synchronization": 0.0, "scheduling": 0.0, "data_movement": 0.0}
        self.stage_busy = defaultdict(float)
        self.stage_wall = {}
        self._shms, self.a, self.pool = [], {}, None

        layout = {
            "X": ((n, d), np.float32),
            "D": ((n, d), np.float32),
            "Z": ((n, k), np.float32),
            "col_min": ((d,), np.float32),
            "span": ((d,), np.float32),
            "mean": ((d,), np.float32),
            "V": ((d, k), np.float32),
            "gram": ((workers, d, d), np.float64),
            "colsum": ((workers, d), np.float64),
        }
        specs = {}
        with timer() as t:
            for key, (shape, dtype) in layout.items():
                nbytes = max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shms.append(shm)
                self.a[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                specs[key] = (shm.name, shape, np.dtype(dtype).str)
            self.a["gram"][:] = 0.0
            self.a["colsum"][:] = 0.0
        self.timing["data_movement"] += t.elapsed

        edges = np.linspace(0, n, min(n, workers * chunks_per_worker) + 1, dtype=int)
        self.spans = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        barrier = ctx.Barrier(workers + 1)
        with timer() as t:
            self.pool = ctx.Pool(workers, initializer=_init_worker,
                                 initargs=(str(raw_path), specs, side, counter, barrier))
            barrier.wait(timeout=120)
        self.timing["scheduling"] += t.elapsed

    def _map(self, fn):
        t0 = time.perf_counter()
        res = self.pool.map(fn, self.spans, chunksize=1)
        wall = time.perf_counter() - t0
        per_pid = defaultdict(float)
        for pid, busy, _ in res:
            for stage, sec in busy.items():
                self.stage_busy[stage] += sec
                per_pid[pid] += sec
        self.timing["communication"] += max(wall - max(per_pid.values()), 0.0)
        return [r[2] for r in res]

    def _main(self, stage, fn, *args, sync=False):
        t0 = time.perf_counter()
        out = fn(*args)
        elapsed = time.perf_counter() - t0
        self.stage_busy[stage] += elapsed
        if sync:
            self.timing["synchronization"] += elapsed
        return out

    def _reduce_minmax(self, partials):
        self.a["col_min"][:], self.a["span"][:] = S.minmax_reduce(partials)

    def _reduce_gram(self):
        return self.a["gram"].sum(axis=0), self.a["colsum"].sum(axis=0)

    def _fit(self, gram, colsum):
        self.a["mean"][:], self.a["V"][:] = S.pca_fit(gram, colsum, self.n, self.k)

    def run_stage_level(self):
        with timer() as t:
            self._map(_task_load)
        self.stage_wall["load"] = t.elapsed

        with timer() as t:
            partials = self._map(_task_minmax)
            self._main("normalize", self._reduce_minmax, partials, sync=True)
            self._map(_task_normalize)
        self.stage_wall["normalize"] = t.elapsed

        with timer() as t:
            self._map(_task_deskew)
        self.stage_wall["deskew"] = t.elapsed

        with timer() as t:
            self._map(_task_gram)
            gram, colsum = self._main("pca", self._reduce_gram, sync=True)
            self._main("pca", self._fit, gram, colsum)
            self._map(_task_project)
        self.stage_wall["pca"] = t.elapsed

    def run_pipeline(self):
        partials = self._map(_task_load_stats)
        self._main("normalize", self._reduce_minmax, partials, sync=True)
        self._map(_task_normalize_deskew_gram)
        gram, colsum = self._main("pca", self._reduce_gram, sync=True)
        self._main("pca", self._fit, gram, colsum)
        self._map(_task_project)

    def collect(self):
        with timer() as t:
            normalized = self.a["X"].copy()
            features = self.a["Z"].copy()
        self.timing["data_movement"] += t.elapsed
        return normalized, features

    def close(self):
        if self.pool is not None:
            with timer() as t:
                self.pool.close()
                self.pool.join()
            self.timing["scheduling"] += t.elapsed
            self.pool = None
        with timer() as t:
            self.a.clear()
            for shm in self._shms:
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
            self._shms = []
        self.timing["data_movement"] += t.elapsed

    def abort(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None
        self.close()


def preprocess_parallel(raw_path, n, n_components, workers, strategy, chunks_per_worker=4):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    run = _ParallelRun(raw_path, n, n_components, workers, chunks_per_worker)
    try:
        if strategy == "stage":
            run.run_stage_level()
        else:
            run.run_pipeline()
        normalized, features = run.collect()
    except BaseException:
        run.abort()
        raise
    run.close()
    return PreprocResult(
        normalized=normalized,
        features=features,
        stage_wall=dict(run.stage_wall),
        stage_busy=dict(run.stage_busy),
        timing=dict(run.timing),
    )
