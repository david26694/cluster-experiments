from typing import List

import numpy as np
import pandas as pd

from cluster_experiments.random_splitter import RandomSplitter

TARGETS = {
    "binary": lambda x: np.random.choice([0, 1], size=x),
    "continuous": lambda x: np.random.normal(0, 1, x),
}


def combine_columns(df, cols_list):
    # combines columns for testing the stratified splitter in case of multiple strata or clusters
    if len(cols_list) > 1:
        return df[cols_list].agg("-".join, axis=1)
    else:
        return df[cols_list]


def assert_balanced_strata(
    splitter: RandomSplitter,
    df: pd.DataFrame,
    strata_cols: List[str],
    cluster_cols: List[str],
    treatments: List[str],
):
    # asserts the balance of the stratified splitter for a given input data frame
    treatment_df = splitter.assign_treatment_df(df)

    treatment_df_unique = treatment_df[
        strata_cols + cluster_cols + ["treatment"]
    ].drop_duplicates()

    treatment_df_unique["clusters_test"] = combine_columns(
        treatment_df_unique, cluster_cols
    )
    treatment_df_unique["strata_test"] = combine_columns(
        treatment_df_unique, strata_cols
    )

    for treatment in treatments:
        for stratus in treatment_df_unique["strata_test"].unique():
            assert (
                treatment_df_unique.query(f"strata_test == '{stratus}'")["treatment"]
                .value_counts(normalize=True)[treatment]
                .squeeze()
            ) == 1 / len(treatments)


def generate_random_data(clusters, dates, N, target="continuous"):
    # Generate random data with clusters and target
    users = [f"User {i}" for i in range(1000)]

    target_values = TARGETS[target](N)
    df = pd.DataFrame(
        {
            "cluster": np.random.choice(clusters, size=N),
            "target": target_values,
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


def generate_clustered_data() -> pd.DataFrame:
    analysis_df = pd.DataFrame(
        {
            "country_code": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "city_code": ["BCN", "BCN", "MAD", "BCN"]
            + ["NAP"] * 4
            + ["WAW"] * 4
            + ["BUC"] * 4,
            "user_id": [1, 1, 2, 1, 3, 4, 5, 6, 7, 8, 8, 8, 9, 9, 9, 10],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"] * 4,
            "treatment": [
                "A",
                "A",
                "B",
                "A",
                "B",
                "B",
                "A",
                "B",
                "B",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "A",
            ],  # Randomization is done at user level, so same user will always have same treatment
            "target": [0.01] * 15 + [0.1],
        }
    )
    return analysis_df
