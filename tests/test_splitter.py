from typing import Counter

import pandas as pd
import pytest
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    ClusteredSplitter,
)


@pytest.fixture
def clusters():
    return ["cluster1", "cluster2", "cluster3", "cluster4"]


@pytest.fixture
def treatments():
    return ["A", "B"]


@pytest.fixture
def dates():
    return ["2022-01-01", "2022-01-02"]


@pytest.fixture
def df_cluster(clusters):
    return pd.DataFrame(
        {
            "cluster": clusters,
        }
    )


@pytest.fixture
def df_switchback(clusters, dates):
    return pd.DataFrame({"cluster": sorted(clusters * 2), "date": dates * 4})


def test_clustered_splitter(clusters, treatments, df_cluster):

    splitter = ClusteredSplitter(cluster_cols=["cluster"], treatments=treatments)
    df_cluster = pd.concat([df_cluster for _ in range(100)])
    sampled_treatment = splitter.assign_treatment_df(df_cluster)
    assert len(sampled_treatment) == len(df_cluster)


def test_balanced_splitter(clusters, treatments, df_cluster):
    splitter = BalancedClusteredSplitter(clusters, treatments)
    sampled_treatment = splitter.sample_treatment(df_cluster)
    assert sorted(sampled_treatment) == ["A", "A", "B", "B"]


def test_balanced_splitter_abc(clusters, df_cluster):
    treatments = ["A", "B", "C"]
    splitter = BalancedClusteredSplitter(clusters, treatments)
    sampled_treatment = splitter.sample_treatment(df_cluster)
    assert max(Counter(sampled_treatment).values()) == 2


def test_switchback_splitter(treatments, df_switchback):
    splitter = ClusteredSplitter(
        cluster_cols=["cluster", "date"], treatments=treatments
    )
    df_switchback_large = pd.concat([df_switchback, df_switchback, df_switchback])
    assignments = splitter.assign_treatment_df(df_switchback_large)
    assert len(assignments) > 0
    assert len(assignments) == len(df_switchback_large)

    # check that each cluster has the same treatment on each date
    assert (assignments.groupby(["date", "cluster"])["treatment"].nunique() == 1).all()


def test_switchback_balanced_splitter(treatments, df_switchback):
    splitter = BalancedClusteredSplitter(
        cluster_cols=["cluster", "date"], treatments=treatments
    )

    df_switchback_large = pd.concat([df_switchback, df_switchback])
    assignments = splitter.assign_treatment_df(df_switchback_large)

    # check that each cluster has the same treatment on each date
    assert (assignments.groupby(["date", "cluster"])["treatment"].nunique() == 1).all()

    # check for treatment balance
    assert (assignments.treatment.value_counts() == 8).all()

    # all treatments are A and B
    assert set(assignments["treatment"]) == set(treatments)


def test_switchback_balanced_splitter_abc():
    df = pd.DataFrame(
        {
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"],
        }
    )
    treatments = ["A", "B", "C"]
    for _ in range(100):
        splitter = BalancedClusteredSplitter(
            cluster_cols=["date"], treatments=treatments
        )
        sampled_treatment = splitter.assign_treatment_df(df)

        counts = sampled_treatment.treatment.value_counts()
        assert counts.max() == 2
        assert counts.min() == 1
