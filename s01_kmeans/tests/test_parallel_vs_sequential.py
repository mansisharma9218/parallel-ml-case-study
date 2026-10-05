"""Correctness check for the parallel implementation: for worker counts
1, 2, 4 it must reach the same centroids as the sequential baseline when
started from identical initial centroids.
"""

import numpy as np
import pytest

from s01_kmeans.data_loading import generate_blobs
from s01_kmeans.parallel import kmeans_parallel
from s01_kmeans.sequential import init_centroids, kmeans_sequential


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_parallel_matches_sequential(n_workers):
    X = generate_blobs(n_samples=4000, n_features=6, centers=6, std=1.0, seed=3)
    k = 6
    initial = init_centroids(X, k, seed=3)

    seq_centroids, _, _ = kmeans_sequential(X, k, initial, max_iter=200, tol=1e-8)
    result = kmeans_parallel(X, k, initial, n_workers=n_workers, max_iter=200, tol=1e-8)

    seq_sorted = np.sort(seq_centroids, axis=0)
    par_sorted = np.sort(result.centroids, axis=0)
    assert np.allclose(seq_sorted, par_sorted, atol=1e-6)


def test_parallel_timing_breakdown_present():
    X = generate_blobs(n_samples=2000, n_features=4, centers=4, std=1.0, seed=9)
    k = 4
    initial = init_centroids(X, k, seed=9)
    result = kmeans_parallel(X, k, initial, n_workers=2, max_iter=50, tol=1e-6)

    for key in ("scheduling", "communication", "synchronization", "compute"):
        assert key in result.timing
        assert result.timing[key] >= 0.0
    assert result.n_iterations >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
