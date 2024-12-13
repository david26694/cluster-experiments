from itertools import product

import numpy as np
import pandas as pd

from ab_lab import ConstantPerturbator, PowerAnalysis
from ab_lab.experiment_analysis import ClusteredOLSAnalysis
from ab_lab.random_splitter import (
    FixedSizeClusteredSplitter,
)


def test_predefined_treatment_clusters_splitter():
    # Create a DataFrame with mock data
    df = pd.DataFrame({"cluster": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]})

    split = FixedSizeClusteredSplitter(cluster_cols=["cluster"], n_treatment_clusters=1)

    df = split.assign_treatment_df(df)

    # Verify that the treatments were assigned correctly
    assert df[split.treatment_col].value_counts()[split.treatments[0]] == 8
    assert df[split.treatment_col].value_counts()[split.treatments[1]] == 2


def test_sample_treatment_with_balanced_clusters():
    splitter = FixedSizeClusteredSplitter(cluster_cols=["city"], n_treatment_clusters=2)
    df = pd.DataFrame({"city": ["A", "B", "C", "D"]})
    treatments = splitter.sample_treatment(df)
    assert len(treatments) == len(df)
    assert treatments.count("A") == 2
    assert treatments.count("B") == 2


def generate_data(N, start_date, end_date):
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    combinations = list(product(users, dates))

    target_values = np.random.normal(100, 10, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    return df


def test_ols_fixed_size_treatment():
    df = generate_data(100, "2021-01-01", "2021-01-15")

    analysis = ClusteredOLSAnalysis(cluster_cols=["user"])

    sw = FixedSizeClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=0,
    )

    pw = PowerAnalysis(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=200
    )
    pw.power_analysis(df, average_effect=0)
    # todo finish this test, the power shouldn't be too high
