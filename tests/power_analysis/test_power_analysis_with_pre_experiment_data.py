from itertools import product

import numpy as np
import pandas as pd

from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysisWithPreExperimentData
from cluster_experiments.random_splitter import FixedSizeClusteredSplitter


def generate_data(N, start_date, end_date):
    # Generate a list of dates between start_date and end_date
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Use itertools.product to create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(100, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    # Ensure 'date' column is of datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df


def test_power_analysis_with_pre_experiment_data():
    df = generate_data(10, "2022-01-01", "2022-01-30")

    sw = FixedSizeClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=0.3,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    )

    pw = PowerAnalysisWithPreExperimentData(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    power = pw.power_analysis(df)
    pw.power_line(df, average_effects=[0.3, 0.4])
    assert 0 <= power <= 1
    values = list(pw.power_line(df, average_effects=[0.3, 0.4]).values())
    assert all(0 <= value <= 1 for value in values)


def test_simulate_point_estimate():
    df = generate_data(10, "2022-01-01", "2022-01-30")

    sw = FixedSizeClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=10,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    )

    pw = PowerAnalysisWithPreExperimentData(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    point_estimates = list(pw.simulate_point_estimate(df))
    assert (
        8 <= pd.Series(point_estimates).mean() <= 11
    ), "Point estimate is too far from the real effect."
