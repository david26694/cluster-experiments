from typing import Counter

import pandas as pd
import pytest
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    SwitchbackSplitter,
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


@pytest.mark.unit
def test_clustered_splitter(clusters, treatments):

    splitter = ClusteredSplitter(clusters, treatments)
    sampled_treatment = splitter.sample_treatment()
    assert len(sampled_treatment) == len(clusters)

    treatment_assignment = splitter.treatment_assignment(sampled_treatment)

    assert set(assignment["cluster"] for assignment in treatment_assignment) == set(
        clusters
    )

    assert sorted(
        [assignment["treatment"] for assignment in treatment_assignment]
    ) == sorted(sampled_treatment)


@pytest.mark.unit
def test_balanced_splitter(clusters, treatments):
    splitter = BalancedClusteredSplitter(clusters, treatments)
    sampled_treatment = splitter.sample_treatment()
    assert sorted(sampled_treatment) == ["A", "A", "B", "B"]


@pytest.mark.unit
def test_balanced_splitter_abc(clusters):
    treatments = ["A", "B", "C"]
    splitter = BalancedClusteredSplitter(clusters, treatments)
    sampled_treatment = splitter.sample_treatment()
    assert max(Counter(sampled_treatment).values()) == 2


@pytest.mark.unit
def test_switchback_splitter(clusters, treatments, dates):
    splitter = SwitchbackSplitter(clusters, treatments, dates)
    sampled_treatment = splitter.sample_treatment()
    assignments = splitter.treatment_assignment(sampled_treatment)
    assert len(assignments) > 0
    assert len(assignments) == len(dates) * len(clusters)

    for assignment in assignments:
        assert assignment["date"] in dates
        assert assignment["cluster"] in clusters
        assert assignment["treatment"] in treatments

    assert (
        pd.DataFrame(assignments)
        .groupby(["date", "cluster"], as_index=False)
        .value_counts()["count"]
        == 1
    ).all()


@pytest.mark.unit
def test_switchback_balanced_splitter(clusters, treatments, dates):
    splitter = BalancedSwitchbackSplitter(clusters, treatments, dates)

    sampled_treatment = splitter.sample_treatment()
    assert list(Counter(sampled_treatment).values()) == [4, 4]
    assert set(Counter(sampled_treatment).keys()) == set(["A", "B"])


@pytest.mark.unit
def test_switchback_balanced_splitter_abc(clusters, dates):
    # TODO: This fails often, balanced switchback not correct
    treatments = ["A", "B", "C"]
    for _ in range(100):
        splitter = BalancedSwitchbackSplitter(clusters, treatments, dates)
        sampled_treatment = splitter.sample_treatment()
        assert max(Counter(sampled_treatment).values()) == 3
        assert min(Counter(sampled_treatment).values()) == 2


@pytest.mark.unit
def test_agg_df(clusters, treatments, df_cluster):
    splitter = ClusteredSplitter(clusters, treatments)
    treatment_df = splitter.assign_treatment_df(df_cluster)
    assert (treatment_df["cluster"] == pd.Series(clusters)).all()


@pytest.mark.unit
def test_agg_df_switchback(clusters, treatments, dates, df_switchback):
    splitter = SwitchbackSplitter(clusters, treatments, dates)
    treatment_df = splitter.assign_treatment_df(df_switchback)
    assert (treatment_df.drop(columns=["treatment"]) == df_switchback).all().all()
