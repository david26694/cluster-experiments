import numpy as np
import pandas as pd
import pytest
from cluster_experiments.experiment_analysis import (
    GeeExperimentAnalysis,
    TTestClusteredAnalysis,
)

from tests.examples import analysis_df, generate_random_data


@pytest.fixture
def analysis_df_diff():
    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02"] * 8,
            "treatment": (["A"] * 4 + ["B"] * 4) * 2,
            "target": [0] * 16,
        }
    )
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analysis_df_full.loc[analysis_df_full["treatment"] == "B", "target"] = 0.1
    return analysis_df_full


def test_cluster_column():
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert (analyser._get_cluster_column(analysis_df) == "Cluster 12022-01-01").all()


def test_binary_treatment():
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert (
        analyser._create_binary_treatment(analysis_df)["treatment"]
        == pd.Series([0, 1, 1, 0])
    ).all()


def test_get_pvalue():
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analyser = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )
    assert analyser.get_pvalue(analysis_df_full) >= 0


def test_ttest(analysis_df_diff):

    analyser = TTestClusteredAnalysis(cluster_cols=["cluster"])

    assert 0.05 >= analyser.get_pvalue(analysis_df_diff) >= 0


def test_ttest_random_data():
    N = 1000
    analysis_df = generate_random_data(
        clusters=[f"c_{i}" for i in range(100)], dates=["2021-01-01"], N=N
    ).assign(treatment=np.random.choice(["A", "B"], size=N))
    analyser = TTestClusteredAnalysis(cluster_cols=["cluster", "date"])

    assert analyser.get_pvalue(analysis_df) >= 0
