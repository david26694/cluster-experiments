# %%
import time
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


# %%
clusters = [f"Cluster {i}" for i in range(1000)]
dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
N = 1_000_000
df = generate_random_data(clusters, dates, N)
sw = ClusteredSplitter(
    cluster_cols=["cluster", "date"],
)

perturbator = ConstantPerturbator(
    average_effect=0.1,
)

analysis = GeeExperimentAnalysis(
    cluster_cols=["cluster", "date"],
)

pw = PowerAnalysis(perturbator=perturbator, splitter=sw, analysis=analysis)

print(df)
# %%

if __name__ == "__main__":

    n_simulations = 16
    n_jobs = 8
    # parallel_start = time.time()
    # parallel_sim = pw.power_analysis_parallel(
    #     df=df, n_simulations=n_simulations, average_effect=-0.01, n_jobs=16
    # )
    # parallel_end = time.time()
    # print("Parallel execution finished")
    # parallel_duration = parallel_end - parallel_start
    # print(f"{parallel_duration=}")

    non_parallel_start = time.time()
    simple_sim = pw.power_analysis(
        df=df, n_simulations=n_simulations, average_effect=-0.01
    )
    non_parallel_end = time.time()
    print("Non Parallel execution finished")
    non_parallel_duration = non_parallel_end - non_parallel_start
    print(f"{non_parallel_duration=}")

    parallel_start = time.time()
    parallel_sim = pw.power_analysis(
        df=df,
        n_simulations=n_simulations,
        average_effect=-0.01,
        n_jobs=n_jobs,
    )
    parallel_end = time.time()
    print("Parallel mp execution finished")
    parallel_duration = parallel_end - parallel_start
    print(f"{parallel_duration=}")
