from datetime import date

import numpy as np
import pytest

from tests.utils import generate_ratio_metric_data


@pytest.fixture
def experiment_dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]


@pytest.fixture
def delta_df(experiment_dates):
    user_sample_mean = 0.3
    user_standard_error = 0.15
    users = 2000
    N = 50_000

    user_target_means = np.random.normal(user_sample_mean, user_standard_error, users)

    data = generate_ratio_metric_data(
        experiment_dates, N, user_target_means, users, treatment_effect=0
    )
    return data
