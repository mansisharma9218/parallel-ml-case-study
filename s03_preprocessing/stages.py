"""Kernels for the four preprocessing stages. The sequential baseline and
both parallel strategies call exactly these functions, so all three do
identical arithmetic and differ only in how the work is split and
scheduled.

  load       raw uint8 rows from a memory-mapped .npy -> float32
             (I/O + memory-bound)
  normalize  per-feature min-max scaling to [0, 1]; needs the global
             per-column min/max, i.e. a reduction over all rows before
             any row can be scaled (memory-bound, one global barrier)
  deskew     moment-based shear correction of each image; independent
             per image and compute-heavy
  pca        covariance via per-chunk Gram matrices (a global reduction),
             a serial eigendecomposition, then per-row projection onto
             the top components
"""

from . import _pin_blas  # noqa: F401  (must precede `import numpy`)

import numpy as np
from scipy.ndimage import affine_transform

STAGES = ["load", "normalize", "deskew", "pca"]

# Rows per inner block inside a kernel: bounds the size of float64
# temporaries independently of how large a chunk the caller passes in.
BLOCK_ROWS = 1024


def load_into(raw_path, start, stop, out):
    """out[start:stop] = rows start..stop of the raw uint8 file, as float32."""
    raw = np.load(raw_path, mmap_mode="r")
    out[start:stop] = raw[start:stop]


def minmax_partial(X):
    return X.min(axis=0), X.max(axis=0)


def minmax_reduce(partials):
    """Combines per-chunk (min, max) pairs into (col_min, span). Same
    formula as S01's fallback normalization, so S01 gets bit-identical
    input whichever of the two sources it loads.
    """
    col_min = np.min([p[0] for p in partials], axis=0).astype(np.float32)
    col_max = np.max([p[1] for p in partials], axis=0).astype(np.float32)
    span = np.where(col_max > col_min, col_max - col_min, 1.0).astype(np.float32)
    return col_min, span


def normalize_inplace(X, col_min, span):
    X -= col_min
    X /= span


def deskew_into(X, out, side):
    """out[i] = image i of X (side x side, flattened) with its shear
    removed and its centre of mass moved to the image centre.
    """
    for b in range(0, len(X), BLOCK_ROWS):
        _deskew_block(X[b:b + BLOCK_ROWS], out[b:b + BLOCK_ROWS], side)


def _deskew_block(X, out, side):
    imgs = X.reshape(-1, side, side)
    out_imgs = out.reshape(-1, side, side)
    w = imgs.astype(np.float64)
    rows, cols = np.mgrid[:side, :side].astype(np.float64)

    total = w.sum(axis=(1, 2))
    safe = np.where(total > 0, total, 1.0)
    mr = (w * rows).sum(axis=(1, 2)) / safe
    mc = (w * cols).sum(axis=(1, 2)) / safe
    dr = rows - mr[:, None, None]
    dc = cols - mc[:, None, None]
    var_r = (dr * dr * w).sum(axis=(1, 2)) / safe
    cov_rc = (dr * dc * w).sum(axis=(1, 2)) / safe
    alpha = np.divide(cov_rc, var_r, out=np.zeros_like(cov_rc), where=var_r > 0)

    center = np.full(2, side / 2.0)
    for i in range(len(imgs)):
        if total[i] <= 0:
            out_imgs[i] = imgs[i]
            continue
        # output pixel o samples input at m @ o + offset: shear rows back
        # by alpha and put the centre of mass at the image centre
        m = np.array([[1.0, 0.0], [alpha[i], 1.0]])
        offset = np.array([mr[i], mc[i]]) - m @ center
        affine_transform(imgs[i], m, offset=offset, output=out_imgs[i],
                         order=1, mode="constant", cval=0.0)


def gram_partial(D):
    """(D^T D, column sums) in float64, accumulated block by block."""
    d = D.shape[1]
    gram = np.zeros((d, d))
    colsum = np.zeros(d)
    for b in range(0, len(D), BLOCK_ROWS):
        B = D[b:b + BLOCK_ROWS].astype(np.float64)
        gram += B.T @ B
        colsum += B.sum(axis=0)
    return gram, colsum


def pca_fit(gram, colsum, n, n_components):
    """Top principal axes from the reduced Gram matrix. Serial: this is
    the part of the pipeline no amount of workers can speed up.
    Returns (mean, components) as float32, components shaped (d, k), with
    each axis' sign fixed (largest-|value| entry positive) so results are
    reproducible across strategies.
    """
    mean = colsum / n
    cov = (gram - n * np.outer(mean, mean)) / max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    k = min(n_components, len(eigvals))
    order = np.argsort(eigvals)[::-1][:k]
    V = eigvecs[:, order]
    signs = np.sign(V[np.abs(V).argmax(axis=0), np.arange(k)])
    signs[signs == 0] = 1.0
    V = V * signs
    return mean.astype(np.float32), V.astype(np.float32)


def project_into(D, mean, V, out):
    for b in range(0, len(D), BLOCK_ROWS):
        out[b:b + BLOCK_ROWS] = (D[b:b + BLOCK_ROWS] - mean) @ V
