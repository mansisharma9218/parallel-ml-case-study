"""Correctness check for the sequential baseline: our Lloyd's K-means
must converge to the same centroids as scikit-learn given identical
initial centroids. sklearn is a test-only dependency — the actual
baseline and experiments never import it.
"""

import numpy as np
from sklearn.cluster import KMeans

from s01_kmeans.data_loading import generate_blobs
from s01_kmeans.sequential import init_centroids, kmeans_sequential


def test_matches_sklearn_small_blobs():
    X = generate_blobs(n_samples=2000, n_features=8, centers=5, std=1.0, seed=7)
    k = 5
    initial = init_centroids(X, k, seed=7)

    ours_centroids, ours_labels, n_iter = kmeans_sequential(
        X, k, initial, max_iter=300, tol=1e-8
    )

    sk = KMeans(
        n_clusters=k,
        init=initial.astype(np.float64),
        n_init=1,
        max_iter=300,
        tol=1e-8,
        algorithm="lloyd",
    ).fit(X.astype(np.float64))

    ours_sorted = np.sort(ours_centroids, axis=0)
    sk_sorted = np.sort(sk.cluster_centers_, axis=0)
    assert np.allclose(ours_sorted, sk_sorted, atol=1e-2)
    assert n_iter <= 300


def test_converges_within_max_iter():
    X = generate_blobs(n_samples=1000, n_features=4, centers=3, std=0.5, seed=1)
    k = 3
    initial = init_centroids(X, k, seed=1)
    _, _, n_iter = kmeans_sequential(X, k, initial, max_iter=100, tol=1e-6)
    assert n_iter < 100
