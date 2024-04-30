import pandas as pd

from cluster_experiments.random_splitter import FixedSizeClusteredSplitter


def test_predefined_treatment_clusters_splitter():
    # Create a DataFrame with mock data
    df = pd.DataFrame({"cluster": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]})

    split = FixedSizeClusteredSplitter(cluster_cols=["cluster"], n_treatment_clusters=1)

    df = split.assign_treatment_df(df)

    # Verify that the treatments were assigned correctly
    assert df[split.treatment_col].value_counts()[split.treatments[0]] == 8
    assert df[split.treatment_col].value_counts()[split.treatments[1]] == 2


def test_sample_treatment_with_balanced_clusters():
    splitter = FixedSizeClusteredSplitter(cluster_cols=["city"], n_treatment_clusters=2)
    df = pd.DataFrame({"city": ["A", "B", "C", "D"]})
    treatments = splitter.sample_treatment(df)
    assert len(treatments) == len(df)
    assert treatments.count("A") == 2
    assert treatments.count("B") == 2
