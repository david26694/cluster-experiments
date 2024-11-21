import numpy as np
import pytest

from cluster_experiments.synthetic_control_utils import loss_w
from cluster_experiments.utils import _get_mapping_key


def test_get_mapping_key():
    with pytest.raises(KeyError):
        mapping = {"a": 1, "b": 2}
        _get_mapping_key(mapping, "c")


def test_loss_w():
    W = np.array([2, 1])  # Weights vector
    X = np.array([[1, 2], [3, 4], [5, 6]])  # Input matrix
    y = np.array([6, 14, 22])  # Actual outputs

    # Calculate expected result
    # Predictions are calculated as follows:
    # [1*2 + 2*1, 3*2 + 4*1, 5*2 + 6*1] = [4, 10, 16]
    # RMSE is sqrt(mean([(6-4)^2, (14-10)^2, (22-16)^2]))
    expected_rmse = np.sqrt(
        np.mean((np.array([6, 14, 22]) - np.array([4, 10, 16])) ** 2)
    )

    # Call the function
    calculated_rmse = loss_w(W, X, y)

    # Assert if the calculated RMSE matches the expected RMSE
    assert np.isclose(
        calculated_rmse, expected_rmse
    ), f"Expected RMSE: {expected_rmse}, but got: {calculated_rmse}"
