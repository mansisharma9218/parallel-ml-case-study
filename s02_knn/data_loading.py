"""MNIST (with labels) and a tiny offline fallback for --quick runs."""
from . import _pin_blas  # noqa: F401
from pathlib import Path

import numpy as np

CACHE = Path("data/raw/mnist_s02.npz")


def load_mnist():
    """Returns (X float32 in [0,1], y int64). Cached after the first download.
    k-NN needs labels, so this does not use S03's normalized .npy (X only).
    """
    if CACHE.exists():
        d = np.load(CACHE)
        return d["X"], d["y"]
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = (X.astype(np.float32) / 255.0)
    y = y.astype(np.int64)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, y=y)
    return X, y


def load_digits(seed=0):
    from sklearn.datasets import load_digits as _ld

    X, y = _ld(return_X_y=True)
    p = np.random.default_rng(seed).permutation(len(X))
    return (X[p] / 16.0).astype(np.float32), y[p].astype(np.int64)


def split_train_query(X, y, n_query_pool, seed):
    """Last n_query_pool rows are the query pool; the rest is training data
    (shuffled with a fixed seed so every size is a random subset)."""
    Xtr, ytr = X[:-n_query_pool], y[:-n_query_pool]
    Xq, yq = X[-n_query_pool:], y[-n_query_pool:]
    p = np.random.default_rng(seed).permutation(len(Xtr))
    c = np.ascontiguousarray
    return c(Xtr[p], dtype=np.float32), ytr[p], c(Xq, dtype=np.float32), yq
