# we want to create a test to confirm the synthetic control analysis works as expected. We will generate a some donors, and 2 of them will be double the target.
# these 2 donors should receive a weight of 0.5 each. The synthetic control should be the average of the 2 donors.

# first we need to generate the data
from itertools import product

import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysisWithPreExperimentData
from cluster_experiments.random_splitter import PredefinedTreatmentClustersSplitter
from cluster_experiments.synthetic_control import get_w


def generate_data(N, start_date, end_date):
    # Generate a list of dates between start_date and end_date
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Use itertools.product to create a combination of each date with each user
    combinations = list(product(users, dates))

    # target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])

    df["target"] = 0
    for i in range(1, N + 1):
        df.loc[df["user"] == f"User {i}", "target"] = i

    # Ensure 'date' column is of datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df


# Example usage


def test_synthetic_control_analysis():
    df = generate_data(10, "2022-01-01", "2022-01-30")

    sw = PredefinedTreatmentClustersSplitter(
        n_treatment_clusters=1, cluster_cols=["user"]
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    )

    pw = PowerAnalysisWithPreExperimentData(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    pw.power_analysis(df)


@pytest.mark.parametrize(
    "X, y, expected_sum, expected_bounds",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([1, 1]),
            1,
            (0, 1),
        ),  # 2D array with positive integers
        (
            np.array([[1, -2], [-3, 4]]),
            np.array([1, 1]),
            1,
            (0, 1),
        ),  # 2D array with negative integers
        (
            np.array([[1.5, 2.5], [3.5, 4.5]]),
            np.array([1, 1]),
            1,
            (0, 1),
        ),  # 2D array with positive floats
        (
            np.array([[1.5, -2.5], [-3.5, 4.5]]),
            np.array([1, 1]),
            1,
            (0, 1),
        ),  # 2D array with negative floats
    ],
)
def test_get_w_weights(X, y, expected_sum, expected_bounds):
    weights = get_w(X, y)
    assert np.isclose(np.sum(weights), expected_sum), "Weights sum should be close to 1"
    assert all(
        expected_bounds[0] <= w <= expected_bounds[1] for w in weights
    ), "Each weight should be between 0 and 1"
