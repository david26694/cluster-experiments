from functools import partial

import numpy as np
from scipy.optimize import fmin_slsqp


def loss_w(W, X, y) -> float:
    return np.sqrt(np.mean((y - X.dot(W)) ** 2))


def get_w(X, y, verbose=False):
    """
    Get weights per unit, constraint in the loss function that sum equals 1 (bounds)
    """
    w_start = np.full(X.shape[1], 1 / X.shape[1])
    bounds = [(0.0, 1.0)] * len(w_start)

    weights = fmin_slsqp(
        partial(loss_w, X=X, y=y),
        w_start,
        f_eqcons=np.sum,
        bounds=bounds,
        disp=verbose,
    )
    return weights
