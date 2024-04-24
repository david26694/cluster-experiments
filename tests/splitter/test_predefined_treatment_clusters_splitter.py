import pandas as pd

from cluster_experiments.random_splitter import FixedSizeClusteredSplitter


def test_predefined_treatment_clusters_splitter():
    # Create a DataFrame with mock data
    df = pd.DataFrame({"cluster": ["A", "B", "C", "D", "E"]})

    # Instantiate PredefinedTreatmentClustersSplitter
    split = FixedSizeClusteredSplitter(cluster_cols=["cluster"], n_treatment_clusters=1)

    df = split.assign_treatment_df(df)

    # Verify that the treatments were assigned correctly
    assert df[split.treatment_col].value_counts()[split.treatments[0]] == 4
    assert df[split.treatment_col].value_counts()[split.treatments[1]] == 1
