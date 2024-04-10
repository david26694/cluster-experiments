from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import fmin_slsqp


def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))


def get_w(X, y):
    """
    Get weights per unit, constraint in the loss function that sum equals 1 (bounds)
    """
    w_start = [1 / X.shape[1]] * X.shape[1]

    weights = fmin_slsqp(
        partial(loss_w, X=X, y=y),
        np.array(w_start),
        f_eqcons=lambda x: np.sum(x) - 1,
        bounds=[(0.0, 1.0)] * len(w_start),
        disp=True,
    )
    return weights


def fit_synthetic(
    df: pd.DataFrame, time_col, cluster_cols, treatment, target_col
) -> list:
    """Returns the fitted OLS model"""

    inverted = df.pivot(index=cluster_cols, columns=time_col)[target_col].T

    y = inverted[treatment].values  # treated
    X = inverted.drop(columns=treatment).values  # donor pool

    weights = get_w(X, y)
    return weights


def synthetic_control(state: int, df: pd.DataFrame, cluster, date) -> np.array:

    weights = fit_synthetic(df)

    synthetic = (
        df.query(f"~(state=={state})")
        .pivot(index="year", columns="state")["cigsale"]
        .values.dot(weights)
    )

    return df.query(f"state=={state}")[
        [cluster, date, "cigsale", "after_treatment"]
    ].assign(synthetic=synthetic)
