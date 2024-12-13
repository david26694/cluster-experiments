from functools import partial
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp


def loss_w(W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    This function calculates the root mean square error (RMSE) between the actual and predicted values in a linear model.
    It is used as an objective function for optimization problems where the goal is to minimize the RMSE.

    Parameters:
    W (numpy.ndarray): The weights vector used for predictions.
    X (numpy.ndarray): The input data matrix.
    y (numpy.ndarray): The actual output vector.

    Returns:
    float: The calculated RMSE.
    """
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))


def get_w(X, y, verbose=False) -> np.ndarray:
    """
    Get weights per unit, constraint in the loss function that sum equals 1; bounds 0 and 1)
    """
    w_start = np.full(X.shape[1], 1 / X.shape[1])
    bounds = [(0.0, 1.0)] * len(w_start)

    weights = fmin_slsqp(
        partial(loss_w, X=X, y=y),
        w_start,
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=bounds,
        disp=verbose,
    )
    return weights


def generate_synthetic_control_data(N, start_date, end_date):
    """Create df for synthetic control cases, where we need a time variable, a target metric, and some clusters."""
    # Generate a list of dates between start_date and end_date
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(100, 10, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    df["date"] = pd.to_datetime(df["date"])

    return df
