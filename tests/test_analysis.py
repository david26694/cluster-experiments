import pandas as pd
from cluster_experiments.experiment_analysis import (
    GeeExperimentAnalysis,
    GeeExperimentAnalysisAggMean,
)

from tests.examples import analysis_df


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
    assert analyser.get_pvalue(analysis_df_full) > 0


def test_agg_mean_covariates():
    analyser = GeeExperimentAnalysisAggMean(cluster_cols=["cluster", "date"])

    assert analyser.covariates == ["target_smooth_mean"]
