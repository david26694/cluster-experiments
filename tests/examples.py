import numpy as np
import pandas as pd


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


def generate_non_clustered_data(N, n_users):
    users = [f"User {i}" for i in range(n_users)]
    df = pd.DataFrame(
        {
            "target": np.random.normal(0, 1, size=N),
            "user": np.random.choice(users, size=N),
        }
    )
    return df


binary_df = pd.DataFrame(
    {
        "target": [0, 1, 0, 1],
        "treatment": ["A", "B", "B", "A"],
    }
)


continuous_df = pd.DataFrame(
    {
        "target": [0.5, 0.5, 0.5, 0.5],
        "treatment": ["A", "B", "B", "A"],
    }
)

analysis_df = pd.DataFrame(
    {
        "target": [0, 1, 0, 1],
        "treatment": ["A", "B", "B", "A"],
        "cluster": ["Cluster 1", "Cluster 1", "Cluster 1", "Cluster 1"],
        "date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
    }
)
