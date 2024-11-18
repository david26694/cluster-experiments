from datetime import date

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


@pytest.fixture
def analysis_ratio_df(dates):
    return generate_ratio_metric_data(dates, N)
