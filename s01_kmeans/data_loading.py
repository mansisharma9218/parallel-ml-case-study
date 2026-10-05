"""Loads the two datasets used by the S01 experiments:

1. Synthetic Gaussian blobs, generated with NumPy at a fixed seed.
2. MNIST, preferring S03's agreed preprocessed output
   (data/processed/mnist_normalized.npy, float32, shape (70000, 784));
   falling back to downloading + min-max normalizing it ourselves if
   that file isn't there yet, so S01 is never blocked waiting on S03.

Everything here stays inside s01_kmeans/; any raw/processed files it
writes go under data/, which is gitignored.
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MNIST_AGREED_PATH = DATA_DIR / "processed" / "mnist_normalized.npy"
MNIST_FALLBACK_PATH = DATA_DIR / "raw" / "mnist_normalized_fallback.npy"
MNIST_EXPECTED_SHAPE = (70000, 784)


def generate_blobs(n_samples, n_features=32, centers=10, std=1.5, seed=42):
    """Synthetic Gaussian blobs: `centers` cluster centers placed on a
    grid-ish random layout, `n_samples` points drawn i.i.d. from
    isotropic Gaussians around them. float32, deterministic given `seed`.
    """
    rng = np.random.RandomState(seed)
    center_points = rng.uniform(low=-50, high=50, size=(centers, n_features))

    counts = np.full(centers, n_samples // centers, dtype=np.int64)
    counts[: n_samples % centers] += 1

    chunks = []
    for center, count in zip(center_points, counts):
        chunks.append(rng.normal(loc=center, scale=std, size=(count, n_features)))
    X = np.concatenate(chunks, axis=0).astype(np.float32)

    perm = rng.permutation(n_samples)
    return X[perm]


def load_mnist(force_fallback=False):
    """Returns (X, source) where X is float32 (70000, 784) in [0, 1] and
    source is "s03" or "fallback".
    """
    if not force_fallback and MNIST_AGREED_PATH.exists():
        X = np.load(MNIST_AGREED_PATH).astype(np.float32, copy=False)
        if X.shape != MNIST_EXPECTED_SHAPE:
            raise ValueError(
                f"{MNIST_AGREED_PATH} has shape {X.shape}, "
                f"expected {MNIST_EXPECTED_SHAPE} per the agreed S03 contract."
            )
        return X, "s03"

    if MNIST_FALLBACK_PATH.exists():
        warnings.warn(
            "Using cached S01 fallback MNIST (S03's "
            f"{MNIST_AGREED_PATH} was not found). Re-run after S03's "
            "preprocessing lands to use the shared version.",
            stacklevel=2,
        )
        return np.load(MNIST_FALLBACK_PATH).astype(np.float32, copy=False), "fallback"

    warnings.warn(
        f"{MNIST_AGREED_PATH} not found (S03's preprocessing hasn't "
        "produced it yet). Falling back to downloading raw MNIST and "
        "min-max normalizing it myself. Results using this path should "
        "be re-checked once S03's output is available.",
        stacklevel=2,
    )
    from sklearn.datasets import fetch_openml

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(raw_dir))
    X = mnist.data.astype(np.float32)

    col_min = X.min(axis=0, keepdims=True)
    col_max = X.max(axis=0, keepdims=True)
    span = np.where(col_max > col_min, col_max - col_min, 1.0)
    X = (X - col_min) / span

    if X.shape != MNIST_EXPECTED_SHAPE:
        raise ValueError(f"Downloaded MNIST has shape {X.shape}, expected {MNIST_EXPECTED_SHAPE}")

    MNIST_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(MNIST_FALLBACK_PATH, X)
    return X, "fallback"
