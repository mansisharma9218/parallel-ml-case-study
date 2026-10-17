"""Raw MNIST for the S03 pipeline, stored as an uncompressed uint8 .npy so
the `load` stage can memory-map it and every worker can read just its own
rows straight from disk.

Downloads via sklearn's fetch_openml into data/raw (the same cache S01's
fallback uses, so it is only downloaded once per machine). Everything
written here lives under data/, which is gitignored.
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_IMAGES_PATH = DATA_DIR / "raw" / "mnist_images_uint8.npy"
RAW_LABELS_PATH = DATA_DIR / "raw" / "mnist_labels_int64.npy"
MNIST_SHAPE = (70000, 784)


def ensure_raw_mnist():
    """Returns (images_path, labels_path), downloading and converting
    MNIST the first time. Images: uint8 (70000, 784), original OpenML
    order (the order S01's fallback also uses).
    """
    if RAW_IMAGES_PATH.exists() and RAW_LABELS_PATH.exists():
        return RAW_IMAGES_PATH, RAW_LABELS_PATH

    from sklearn.datasets import fetch_openml

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, data_home=str(raw_dir))
    data = np.asarray(mnist.data)
    if data.shape != MNIST_SHAPE:
        raise ValueError(f"Downloaded MNIST has shape {data.shape}, expected {MNIST_SHAPE}")
    if data.min() < 0 or data.max() > 255 or not np.array_equal(data, np.round(data)):
        raise ValueError("Downloaded MNIST pixels are not integers in [0, 255]")

    np.save(RAW_IMAGES_PATH, data.astype(np.uint8))
    np.save(RAW_LABELS_PATH, np.asarray(mnist.target).astype(np.int64))
    return RAW_IMAGES_PATH, RAW_LABELS_PATH


def write_raw_npy(images_uint8, path):
    """Writes an arbitrary (n, side*side) uint8 image array in the same
    format, e.g. for tests that must not need network access.
    """
    images_uint8 = np.ascontiguousarray(images_uint8, dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, images_uint8)
    return path
