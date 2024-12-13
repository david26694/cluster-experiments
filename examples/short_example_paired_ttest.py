from datetime import date

import numpy as np
import pandas as pd

from ab_lab.experiment_analysis import PairedTTestClusteredAnalysis
from ab_lab.perturbator import ConstantPerturbator
from ab_lab.power_analysis import PowerAnalysis
from ab_lab.random_splitter import StratifiedSwitchbackSplitter


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

    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.day_name()

    return df


if __name__ == "__main__":
    clusters = [f"Cluster {i}" for i in range(100)]
    dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
    N = 10_000
    df = generate_random_data(clusters, dates, N)
    sw = StratifiedSwitchbackSplitter(
        cluster_cols=["cluster", "date"], strata_cols=["dow"]
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
    )

    analysis = PairedTTestClusteredAnalysis(
        cluster_cols=["cluster", "date"], strata_cols=["cluster"]
    )

    pw = PowerAnalysis(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    print(df)
    power = pw.power_analysis(df)
    print(f"{power = }")
