from itertools import product

import numpy as np
import pandas as pd

from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysisWithPreExperimentData
from cluster_experiments.random_splitter import PredefinedTreatmentClustersSplitter


def generate_data(N, start_date, end_date):
    # Generate a list of dates between start_date and end_date
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Use itertools.product to create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    # Ensure 'date' column is of datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df


# Example usage


def test_power_analysis_with_pre_experiment_data():
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

    power = pw.power_analysis(df)
    assert 0 <= power <= 1
