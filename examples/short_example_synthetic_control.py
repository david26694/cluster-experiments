from datetime import date
from itertools import product

import numpy as np
import pandas as pd

from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter


def generate_data(N, start_date, end_date):
    # Generate a list of dates between start_date and end_date
    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Use itertools.product to create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    # Ensure 'date' column is of datetime type and extract day of week name
    df["date"] = pd.to_datetime(df["date"])

    return df


# Example usage


if __name__ == "__main__":

    df = generate_data(10, "2022-01-01", "2022-01-30")
    df["treatment_period"] = np.where(
        df["date"].dt.date < date(2022, 1, 15), "Before", "After"
    )
    sw = ClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=0.1,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", transition_date=date(2022, 1, 15)
    )

    pw = PowerAnalysis(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    print(df)
    power = pw.power_analysis(df)
    set(pw.simulate_point_estimate(df))
    print(f"{power = }")
