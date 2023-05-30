from datetime import date

import numpy as np
import pandas as pd

from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter


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
    experiment_dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]
    N = 10_000
    df = generate_random_data(clusters, dates, N)
    sw = ClusteredSplitter(
        treatments=["A", "B"],
        cluster_cols=["cluster", "date"],
    )

    treatment_assignment_df = sw.assign_treatment_df(df)
    # NaNs because of data previous to experiment
    print(treatment_assignment_df)

    perturbator = ConstantPerturbator(
        average_effect=0.01,
        target_col="target",
        treatment_col="treatment",
    )

    perturbated_df = perturbator.perturbate(treatment_assignment_df)
    print(perturbated_df.groupby(["treatment"]).mean())

    analysis = GeeExperimentAnalysis(
        target_col="target",
        treatment_col="treatment",
        cluster_cols=["cluster", "date"],
    )

    p_val = analysis.get_pvalue(perturbated_df.query("treatment.notnull()"))
    print(f"{p_val = }")

    pw = PowerAnalysis(
        target_col="target",
        treatment_col="treatment",
        treatment="B",
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
    )

    print(df)
    power = pw.power_analysis(df)
    print(f"{power = }")
