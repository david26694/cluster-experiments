from datetime import date

import numpy as np
import pandas as pd
import pytest

from tests.utils import generate_ratio_metric_data

N = 50_000


@pytest.fixture
def dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]


@pytest.fixture
def experiment_dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]


@pytest.fixture
def analysis_df():
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "treatment": ["A", "B", "B", "A"],
            "cluster": ["Cluster 1", "Cluster 1", "Cluster 1", "Cluster 1"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
        }
    )


def analysis_ratio_df_base(dates, experiment_dates, n_users=2000):
    pre_exp_dates = [d for d in dates if d not in experiment_dates]

    user_sample_mean = 0.3
    user_standard_error = 0.15
    users = n_users

    user_target_means = np.random.normal(user_sample_mean, user_standard_error, users)

    pre_data = generate_ratio_metric_data(
        pre_exp_dates, N, user_target_means, users, treatment_effect=0
    )
    post_data = generate_ratio_metric_data(
        experiment_dates, N, user_target_means, users
    )
    return pd.concat([pre_data, post_data])


@pytest.fixture
def analysis_ratio_df(dates, experiment_dates):
    return analysis_ratio_df_base(dates, experiment_dates, n_users=2000)


@pytest.fixture
def analysis_ratio_df_large(dates, experiment_dates, n_users=200000):
    return analysis_ratio_df_base(dates, experiment_dates, n_users=n_users)


@pytest.fixture
def covariate_data():
    """generates data via y ~ T + X"""
    N = 1000
    np.random.seed(123)
    X = np.random.normal(size=N)
    T = np.random.choice(["A", "B"], size=N)
    y = 0.5 * X + 0.1 * (T == "B") + np.random.normal(size=N)
    df = pd.DataFrame({"y": y, "T": T, "X": X})
    return df
