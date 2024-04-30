from itertools import product

import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.synthetic_control import get_w


def generate_data(N, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="d")
    country = ["US", "UK"]
    users = [f"User {i}" for i in range(N)]

    # Get the combination of each date with each user
    combinations = list(product(users, dates, country))

    df = pd.DataFrame(combinations, columns=["user", "date", "country"])

    df["target"] = np.random.normal(0, 1, size=len(combinations))
    # Ensure 'date' column is of datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df


def test_synthetic_control_analysis():
    df = generate_data(10, "2022-01-01", "2022-01-30")

    # Add treatment column to only 1 user
    df["treatment"] = "A"
    df.loc[(df["user"] == "User 5") & (df["country"] == "US"), "treatment"] = "B"

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user", "country"], intervention_date="2022-01-06"
    )

    p_value = analysis.get_pvalue(df)
    assert 0 <= p_value <= 1


@pytest.mark.parametrize(
    "X, y",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([1, 1]),
        ),  # Scenario with positive integers
        (
            np.array([[1, -2], [-3, 4]]),
            np.array([1, 1]),
        ),  # Scenario with negative integers
        (
            np.array([[1.5, 2.5], [3.5, 4.5]]),
            np.array([1, 1]),
        ),  # Scenario with positive floats
        (
            np.array([[1.5, -2.5], [-3.5, 4.5]]),
            np.array([1, 1]),
        ),  # Scenario with negative floats
    ],
)
def test_get_w_weights(
    X, y
):  # this function is not part of the analysis, but it is used in it
    expected_sum = 1
    expected_bounds = (0, 1)
    weights = get_w(X, y)
    assert np.isclose(np.sum(weights), expected_sum), "Weights sum should be close to 1"
    assert all(
        expected_bounds[0] <= w <= expected_bounds[1] for w in weights
    ), "Each weight should be between 0 and 1"


def test_get_treatment_cluster_returns_correct_cluster():
    analysis = SyntheticControlAnalysis(cluster_cols=["cluster"])
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4, 5, 6],
            "treatment": [0, 0, 1, 1, 1, 0],
            "cluster": [
                "cluster1",
                "cluster2",
                "cluster3",
                "cluster3",
                "cluster3",
                "cluster2",
            ],
        }
    )
    expected_cluster = "cluster3"
    assert analysis._get_treatment_cluster(df) == expected_cluster
