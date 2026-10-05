"""Sequential, vectorized NumPy Lloyd's K-means — the baseline every
speedup number is measured against. No multiprocessing here at all.
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import numpy as np


def init_centroids(X, k, seed):
    """Deterministic initial centroids: k distinct points chosen without
    replacement. Callers pass the same seed to both the sequential and
    parallel runs so both start from identical centroids.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=k, replace=False)
    return X[idx].copy().astype(np.float64)


def grouped_sums(X_T, labels, k):
    """Per-cluster sums of X's rows, grouped by `labels` in [0, k).

    Takes X transposed (shape (d, n)) and does one np.bincount(weights=...)
    per feature column instead of np.add.at, which is much slower at the
    sizes used here. Returns (sums, counts); sums in float64.
    """
    sums = np.stack(
        [np.bincount(labels, weights=col, minlength=k) for col in X_T], axis=1
    )
    counts = np.bincount(labels, minlength=k)
    return sums, counts


def _assign(X, centroids):
    """Nearest-centroid assignment via ||x-c||^2 = ||x||^2 - 2x.c + ||c||^2,
    computed with a matmul so memory stays O(n*k) instead of O(n*k*d).
    """
    centroids32 = centroids.astype(np.float32, copy=False)
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    c_sq = np.einsum("ij,ij->i", centroids32, centroids32)[None, :]
    cross = X @ centroids32.T
    dists = x_sq - 2.0 * cross + c_sq
    return dists.argmin(axis=1)


def kmeans_sequential(X, k, initial_centroids, max_iter=100, tol=1e-4):
    """Lloyd's algorithm. Returns (centroids, labels, n_iterations).

    `initial_centroids` must be provided by the caller (via
    `init_centroids`) so sequential and parallel runs are comparable.
    X is kept float32 (the assignment step runs in float32); centroids
    are accumulated in float64 for numerically stable running updates.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    X_T = np.ascontiguousarray(X.T)  # built once; grouped_sums needs column access every iteration
    centroids = initial_centroids.astype(np.float64).copy()
    labels = None

    for iteration in range(1, max_iter + 1):
        labels = _assign(X, centroids)

        new_centroids = centroids.copy()
        sums, counts = grouped_sums(X_T, labels, k)
        nonempty = counts > 0
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        # empty cluster: keep its previous centroid in place.

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            return centroids, labels, iteration

    return centroids, labels, max_iter
