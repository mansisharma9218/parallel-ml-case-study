"""Correctness of the individual stage kernels."""

import numpy as np

from s03_preprocessing import stages as S


def test_minmax_matches_s01_fallback_formula_exactly():
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, size=(500, 49)).astype(np.float32)
    raw[:, 0] = 7.0  # constant column -> span falls back to 1

    # S01's fallback (s01_kmeans/data_loading.py), applied to the whole array
    col_min = raw.min(axis=0, keepdims=True)
    col_max = raw.max(axis=0, keepdims=True)
    span = np.where(col_max > col_min, col_max - col_min, 1.0)
    expected = (raw - col_min) / span

    X = raw.copy()
    partials = [S.minmax_partial(X[a:b]) for a, b in [(0, 120), (120, 333), (333, 500)]]
    S.normalize_inplace(X, *S.minmax_reduce(partials))

    assert X.dtype == np.float32
    np.testing.assert_array_equal(X, expected)


def _sheared_bar(side=28, alpha=0.5):
    img = np.zeros((side, side), dtype=np.float32)
    for r in range(5, side - 5):
        c = int(round(side / 2 + alpha * (r - side / 2)))
        img[r, c - 1:c + 2] = 1.0
    return img


def _row_centroid_spread(img):
    cols = np.arange(img.shape[1])
    rows_with_ink = img.sum(axis=1) > 0.5
    centroids = (img[rows_with_ink] * cols).sum(axis=1) / img[rows_with_ink].sum(axis=1)
    return centroids.std()


def test_deskew_straightens_a_sheared_stroke():
    img = _sheared_bar()
    out = np.empty((1, img.size), dtype=np.float32)
    S.deskew_into(img.reshape(1, -1), out, 28)
    before = _row_centroid_spread(img)
    after = _row_centroid_spread(out.reshape(28, 28))
    assert before > 2.0
    assert after < 0.5


def test_deskew_is_independent_of_chunking():
    rng = np.random.default_rng(1)
    X = (rng.random((300, 64)) > 0.7).astype(np.float32)
    X[5] = 0.0  # blank image must pass through unchanged
    whole = np.empty_like(X)
    S.deskew_into(X, whole, 8)
    chunked = np.empty_like(X)
    for a, b in [(0, 7), (7, 150), (150, 300)]:
        S.deskew_into(X[a:b], chunked[a:b], 8)
    np.testing.assert_array_equal(whole, chunked)
    np.testing.assert_array_equal(whole[5], 0.0)


def test_pca_matches_svd_subspace():
    rng = np.random.default_rng(2)
    n, d, k = 2000, 30, 4
    Q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    latent = rng.normal(size=(n, k)) * np.array([10.0, 6.0, 3.0, 1.5])
    X = (latent @ Q.T + 0.01 * rng.normal(size=(n, d)) + 5.0).astype(np.float32)

    gram, colsum = S.gram_partial(X)
    mean, V = S.pca_fit(gram, colsum, n, k)

    Xc = X.astype(np.float64) - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    overlap = np.abs(V.T.astype(np.float64) @ Vt[:k].T)
    np.testing.assert_allclose(np.diag(overlap), 1.0, atol=1e-4)
    np.testing.assert_allclose(mean, X.mean(axis=0), rtol=1e-5)

    Z = np.empty((n, k), dtype=np.float32)
    S.project_into(X, mean, V, Z)
    np.testing.assert_allclose(Z.var(axis=0), latent.var(axis=0), rtol=0.05)
