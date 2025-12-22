from datetime import date

import numpy as np
import pandas as pd

from cluster_experiments.cupac import TargetAggregation
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter


def generate_random_data(clusters, dates, N, n_users=1000):
    # Generate random data with clusters and target
    users = [f"User {i}" for i in range(n_users)]

    # Every user has a mean
    df_users = pd.DataFrame(
        {"user": users, "user_mean": np.random.normal(0, 3, size=n_users)}
    )

    # Every cluster has a mean
    df_clusters = pd.DataFrame(
        {
            "cluster": clusters,
            "cluster_mean": np.random.normal(0, 0.1, size=len(clusters)),
        }
    )
    # The target is the sum of: user mean, cluster mean and random residual
    df = (
        pd.DataFrame(
            {
                "cluster": np.random.choice(clusters, size=N),
                "residual": np.random.normal(0, 1, size=N),
                "user": np.random.choice(users, size=N),
                "date": np.random.choice(dates, size=N),
            }
        )
        .merge(df_users, on="user")
        .merge(df_clusters, on="cluster")
        .assign(target=lambda x: x["residual"] + x["user_mean"] + x["cluster_mean"])
    )

    return df


if __name__ == "__main__":
    clusters = [f"Cluster {i}" for i in range(100)]
    dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
    experiment_dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]
    N = 10_000
    df = generate_random_data(clusters, dates, N)
    df_analysis = df.query(f"date.isin({experiment_dates})")
    df_pre = df.query(f"~date.isin({experiment_dates})")
    print(df)

    # Splitter and perturbator
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
    )

    # Vainilla GEE
    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    pw_vanilla = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        n_simulations=50,
    )

    power = pw_vanilla.power_analysis(df_analysis)
    print(f"Not using cupac: {power = }")

    # Cupac GEE
    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"], covariates=["estimate_target"]
    )

    target_agg = TargetAggregation(target_col="target", agg_col="user")
    pw_cupac = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        n_simulations=50,
        cupac_model=target_agg,
    )

    power = pw_cupac.power_analysis(df_analysis, df_pre)
    print(f"Using cupac: {power = }")
