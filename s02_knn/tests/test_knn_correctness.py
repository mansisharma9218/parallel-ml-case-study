"""Parallel strategies must return the same neighbours as the sequential
baseline, and the baseline must agree with scikit-learn."""
import numpy as np
import pytest

from s02_knn.knn import PoolRunner, SearchContext, search_sequential, vote

K = 5


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 32)).astype(np.float32)
    y = rng.integers(0, 10, 2000)
    Q = rng.normal(size=(150, 32)).astype(np.float32)
    xsq = np.einsum("ij,ij->i", X, X).astype(np.float32)
    return X, y, Q, xsq


def _same_sets(a, b):
    return np.mean([set(r1) == set(r2) for r1, r2 in zip(a, b)])


def test_sequential_matches_sklearn(problem):
    from sklearn.neighbors import NearestNeighbors
    X, y, Q, xsq = problem
    ref = NearestNeighbors(n_neighbors=K, algorithm="brute").fit(X).kneighbors(Q, return_distance=False)
    assert _same_sets(search_sequential(Q, X, xsq, K), ref) >= 0.99


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_parallel_matches_sequential(problem, workers):
    X, y, Q, xsq = problem
    seq = search_sequential(Q, X, xsq, K)
    with SearchContext(X, Q, K) as ctx, PoolRunner(ctx, workers) as r:
        qi, qc, qs = r.query_parallel()
        di, dc, ds = r.data_parallel()
        assert r.scheduling_time > 0
    assert _same_sets(qi, seq) >= 0.99 and _same_sets(di, seq) >= 0.99
    assert (vote(y[qi]) == vote(y[seq])).mean() >= 0.99
    assert min(qc, qs, dc, ds) >= 0.0


def test_more_workers_than_queries(problem):
    X, y, Q, xsq = problem
    Q = Q[:3]
    seq = search_sequential(Q, X, xsq, K)
    with SearchContext(X, Q, K) as ctx, PoolRunner(ctx, 4) as r:
        assert _same_sets(r.query_parallel()[0], seq) == 1.0
