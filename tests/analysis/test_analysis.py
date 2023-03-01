import numpy as np
import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import (
    ExperimentAnalysis,
    GeeExperimentAnalysis,
    PairedTTestClusteredAnalysis,
    TTestClusteredAnalysis,
)
from tests.examples import analysis_df, generate_clustered_data, generate_random_data


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


def test_paired_ttest():
    "This test make sure that pivot table works as expected (inside get_pvalue) and that paired t test returns a possible value"

    analysis_df = pd.DataFrame(
        {
            "cluster": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"] * 4,
            "treatment": ["A", "B"] * 8,
            "target": [0.01] * 16,
        }
    )
    # Changing just one observation so we have a p value
    analysis_df.loc[1, "target"] = 0.001

    analyser = PairedTTestClusteredAnalysis(
        cluster_cols=["cluster", "date"], strata_cols=["cluster"]
    )

    assert 1 >= analyser.get_pvalue(analysis_df) >= 0, "p value is null or inf"


def test_paired_ttest_preprocessing():

    analyser = PairedTTestClusteredAnalysis(
        cluster_cols=["country_code", "city_code"], strata_cols=["country_code"]
    )
    df = generate_clustered_data()

    df_pivot = analyser._preprocessing(df=df)

    assert df_pivot.isna().sum().sum() == 0, "Unexpected nas in pivot"
    assert (df_pivot.index == ["ES", "IT", "PL", "RO"]).all(), "wrong index"
    assert (df_pivot.columns == ["A", "B"]).all(), "wrong columns"
    assert (
        df_pivot.values == [[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.1, 0.01]]
    ).all(), "wrong values"

    assert df_pivot.shape == (
        df["country_code"].nunique(),
        2,
    ), "different shape than expected"


def test_ttest_random_data():
    N = 1000
    analysis_df = generate_random_data(
        clusters=[f"c_{i}" for i in range(100)], dates=["2021-01-01"], N=N
    ).assign(treatment=np.random.choice(["A", "B"], size=N))
    analyser = TTestClusteredAnalysis(cluster_cols=["cluster", "date"])

    assert analyser.get_pvalue(analysis_df) >= 0


def test_point_estimate_raises():
    class DummyAnalysis(ExperimentAnalysis):
        def __init__(self):
            pass

        def analysis_pvalue(self, df):
            return 0.05

    analyser = DummyAnalysis()
    with pytest.raises(NotImplementedError):
        analyser.analysis_point_estimate(df=pd.DataFrame())
