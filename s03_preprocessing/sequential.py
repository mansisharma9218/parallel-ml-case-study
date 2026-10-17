"""Baseline: the whole preprocessing pipeline in the main process, stage
after stage, single-threaded (BLAS pinned to 1 thread by _pin_blas).
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import math
from dataclasses import dataclass, field

import numpy as np

from common import timer

from . import stages as S


@dataclass
class PreprocResult:
    normalized: np.ndarray  # (n, d) float32 in [0, 1]: output of `normalize`
    features: np.ndarray  # (n, k) float32: deskewed + PCA-projected
    stage_wall: dict  # stage -> wall seconds; empty if a strategy fuses stages
    stage_busy: dict  # stage -> busy seconds summed over all processes
    timing: dict = field(default_factory=dict)  # communication/synchronization/scheduling/data_movement


def raw_geometry(raw_path, n_components):
    """(rows available, features d, image side, output components k)."""
    raw = np.load(raw_path, mmap_mode="r")
    n_avail, d = raw.shape
    side = math.isqrt(d)
    if side * side != d:
        raise ValueError(f"{raw_path}: {d} features is not a square image")
    return n_avail, d, side, min(n_components, d)


def preprocess_sequential(raw_path, n, n_components):
    n_avail, d, side, k = raw_geometry(raw_path, n_components)
    if n > n_avail:
        raise ValueError(f"Asked for {n} rows but {raw_path} has {n_avail}")
    wall = {}

    with timer() as t:
        X = np.empty((n, d), dtype=np.float32)
        S.load_into(raw_path, 0, n, X)
    wall["load"] = t.elapsed

    with timer() as t:
        col_min, span = S.minmax_reduce([S.minmax_partial(X)])
        S.normalize_inplace(X, col_min, span)
    wall["normalize"] = t.elapsed

    with timer() as t:
        D = np.empty_like(X)
        S.deskew_into(X, D, side)
    wall["deskew"] = t.elapsed

    with timer() as t:
        gram, colsum = S.gram_partial(D)
        mean, V = S.pca_fit(gram, colsum, n, k)
        Z = np.empty((n, k), dtype=np.float32)
        S.project_into(D, mean, V, Z)
    wall["pca"] = t.elapsed

    timing = {"communication": 0.0, "synchronization": 0.0, "scheduling": 0.0, "data_movement": 0.0}
    return PreprocResult(normalized=X, features=Z, stage_wall=wall, stage_busy=dict(wall), timing=timing)
