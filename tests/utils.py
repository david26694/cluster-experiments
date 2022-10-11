from typing import List

import pandas as pd
from cluster_experiments.random_splitter import RandomSplitter


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
