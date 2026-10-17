"""Both parallel strategies must produce the same outputs as the sequential
baseline. Uses sklearn's bundled 8x8 digits, so no network is needed.
"""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from s03_preprocessing.data_loading import write_raw_npy
from s03_preprocessing.parallel import preprocess_parallel
from s03_preprocessing.sequential import preprocess_sequential
from s03_preprocessing.stages import STAGES

N_COMPONENTS = 10


@pytest.fixture(scope="module")
def raw_path(tmp_path_factory):
    X, _ = load_digits(return_X_y=True)  # values 0..16
    return write_raw_npy((X * 15).astype(np.uint8), tmp_path_factory.mktemp("s03") / "digits.npy")


@pytest.fixture(scope="module")
def sequential(raw_path):
    return preprocess_sequential(raw_path, 1500, N_COMPONENTS)


def test_sequential_outputs_are_well_formed(sequential):
    assert sequential.normalized.shape == (1500, 64)
    assert sequential.normalized.dtype == np.float32
    assert sequential.normalized.min() >= 0.0 and sequential.normalized.max() <= 1.0
    assert sequential.features.shape == (1500, N_COMPONENTS)
    assert set(sequential.stage_wall) == set(STAGES)


@pytest.mark.parametrize("strategy", ["stage", "pipeline"])
@pytest.mark.parametrize("workers", [1, 3])
def test_parallel_matches_sequential(raw_path, sequential, strategy, workers):
    result = preprocess_parallel(raw_path, 1500, N_COMPONENTS, workers, strategy)

    np.testing.assert_array_equal(result.normalized, sequential.normalized)
    np.testing.assert_allclose(result.features, sequential.features, atol=1e-4)

    assert set(result.stage_busy) == set(STAGES)
    if strategy == "stage":
        assert set(result.stage_wall) == set(STAGES)
    else:
        assert result.stage_wall == {}
    assert all(v >= 0 for v in result.timing.values())
    assert result.timing["scheduling"] > 0
