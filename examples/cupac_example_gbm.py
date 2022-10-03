from datetime import date

import numpy as np
import pandas as pd
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter
from sklearn.ensemble import HistGradientBoostingRegressor


def generate_random_data(clusters, dates, N):

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
                "date": np.random.choice(dates, size=N),
                "x1": np.random.normal(0, 1, size=N),
                "x2": np.random.normal(0, 1, size=N),
                "x3": np.random.normal(0, 1, size=N),
                "x4": np.random.normal(0, 1, size=N),
            }
        )
        .merge(df_clusters, on="cluster")
        .assign(
            target=lambda x: x["x1"] * x["x2"]
            + x["x3"] ** 2
            + x["x4"]
            + x["cluster_mean"]
            + x["residual"]
        )
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

    perturbator = UniformPerturbator(
        average_effect=0.1,
    )

    # Vainilla GEE
    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    pw_vainilla = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        n_simulations=50,
    )

    power = pw_vainilla.power_analysis(df_analysis)
    print(f"Not using cupac: {power = }")

    # Cupac GEE
    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"], covariates=["estimate_target"]
    )

    gbm = HistGradientBoostingRegressor()
    pw_cupac = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        n_simulations=50,
        cupac_model=gbm,
        features_cupac_model=["x1", "x2", "x3", "x4"],
    )

    power = pw_cupac.power_analysis(df_analysis, df_pre)
    print(f"Using cupac: {power = }")
