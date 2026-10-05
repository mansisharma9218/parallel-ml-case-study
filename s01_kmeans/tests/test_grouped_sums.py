"""Correctness check for grouped_sums (the bincount-per-column reduction
that replaced an initial, much slower np.add.at implementation): results
must match a naive per-cluster loop exactly (up to float64 summation
order).
"""

import numpy as np

from s01_kmeans.sequential import grouped_sums


def _naive_grouped_sums(X, labels, k):
    sums = np.zeros((k, X.shape[1]), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for j in range(k):
        mask = labels == j
        counts[j] = mask.sum()
        if counts[j] > 0:
            sums[j] = X[mask].sum(axis=0, dtype=np.float64)
    return sums, counts


def test_matches_naive_reduction():
    rng = np.random.RandomState(0)
    n, d, k = 5000, 10, 7
    X = rng.rand(n, d).astype(np.float32)
    labels = rng.randint(0, k, size=n)

    expected_sums, expected_counts = _naive_grouped_sums(X, labels, k)
    sums, counts = grouped_sums(X.T, labels, k)

    assert np.array_equal(counts, expected_counts)
    assert np.allclose(sums, expected_sums, atol=1e-3)


def test_handles_empty_clusters():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    labels = np.array([0, 0])  # cluster 1 is empty
    sums, counts = grouped_sums(X.T, labels, k=2)

    assert counts.tolist() == [2, 0]
    assert np.allclose(sums[0], [4.0, 6.0])
    assert np.allclose(sums[1], [0.0, 0.0])
