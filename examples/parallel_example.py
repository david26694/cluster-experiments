#%%
import time
from datetime import date

import numpy as np
import pandas as pd

from cluster_experiments.experiment_analysis import OLSAnalysis
from cluster_experiments.perturbator import UniformPerturbator

# from cluster_experiments.power_analysis_parallel import PowerAnalysis as ParallelPowerAnalysis
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import NonClusteredSplitter


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
clusters = [f"Cluster {i}" for i in range(100)]
dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
N = 1000
df = generate_random_data(clusters, dates, N)
sw = NonClusteredSplitter()

perturbator = UniformPerturbator(
    average_effect=0.1,
)

analysis = OLSAnalysis()

pw = PowerAnalysis(
    perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
)

print(df)
# %%
parallel_start = time.time()
parallel_sim = pw.simulate_pvalue(
    df=df, n_simulations=100, average_effect=-0.01, n_processes=10
)
parallel_end = time.time()
print("Parallel execution finished")
parallel_duration = parallel_end - parallel_start
print(f"{parallel_duration=}")

# %%
non_parallel_start = time.time()
simple_sim = list(pw.simulate_pvalue(df=df, n_simulations=100, average_effect=-0.01))
non_parallel_end = time.time()
print("Non Parallel execution finished")
non_parallel_duration = non_parallel_end - non_parallel_start
print(f"{non_parallel_duration=}")

# %%
