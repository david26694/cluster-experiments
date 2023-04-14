from datetime import date

import numpy as np
import pandas as pd

from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.power_config import PowerConfig


def generate_random_data(clusters, dates, N):
    # Generate random data with clusters and target
    users = [f"User {i}" for i in range(1000)]
    df = pd.DataFrame(
        {
            "cluster": np.random.choice(clusters, size=N),
            "target": np.random.normal(0, 1, size=N),
            "user": np.random.choice(users, size=N),
            "date": np.random.choice(dates, size=N),
        }
    )

    return df


if __name__ == "__main__":
    clusters = [f"Cluster {i}" for i in range(100)]
    dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
    N = 10_000
    df = generate_random_data(clusters, dates, N)
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="uniform",
        splitter="clustered",
        n_simulations=100,
    )
    pw = PowerAnalysis.from_config(config)

    print(df)
    power = pw.power_analysis(df)
    print(f"{power = }")
