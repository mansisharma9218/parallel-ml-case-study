"""Produces the preprocessed MNIST files the other sub-problems consume:

  data/processed/mnist_normalized.npy      float32 (70000, 784) in [0, 1]
      the agreed S01 input; bit-identical to S01's own fallback
      normalization, so S01's results don't change when it switches over
  data/processed/mnist_labels.npy          int64 (70000,)
      labels in the same row order (S02's k-NN needs labels)
  data/processed/mnist_deskewed_pca50.npy  float32 (70000, 50)
      output of the full pipeline (normalize -> deskew -> PCA)

Usage: python -m s03_preprocessing.preprocess [--workers N]
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import argparse
import os
import shutil

import numpy as np

from . import config
from .data_loading import DATA_DIR, MNIST_SHAPE, ensure_raw_mnist
from .parallel import preprocess_parallel

PROCESSED_DIR = DATA_DIR / "processed"


def main():
    parser = argparse.ArgumentParser(description="Write S03's preprocessed MNIST outputs.")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    images_path, labels_path = ensure_raw_mnist()
    result = preprocess_parallel(images_path, MNIST_SHAPE[0], config.N_COMPONENTS, args.workers,
                                 "stage", config.CHUNKS_PER_WORKER)

    X = result.normalized
    if X.shape != MNIST_SHAPE or X.dtype != np.float32 or X.min() < 0.0 or X.max() > 1.0:
        raise ValueError(f"Unexpected normalized output: shape {X.shape}, dtype {X.dtype}, "
                         f"range [{X.min()}, {X.max()}]")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "mnist_normalized.npy", X)
    shutil.copyfile(labels_path, PROCESSED_DIR / "mnist_labels.npy")
    np.save(PROCESSED_DIR / f"mnist_deskewed_pca{config.N_COMPONENTS}.npy", result.features)
    print(f"Wrote {PROCESSED_DIR} with {args.workers} workers "
          f"(stage times: { {k: round(v, 3) for k, v in result.stage_wall.items()} })")


if __name__ == "__main__":
    main()
