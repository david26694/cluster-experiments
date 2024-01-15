import pandas as pd

from cluster_experiments.experiment_analysis import OLSAnalysis


def test_binary_treatment(analysis_df):
    analyser = OLSAnalysis()
    assert (
        analyser._create_binary_treatment(analysis_df)["treatment"]
        == pd.Series([0, 1, 1, 0])
    ).all()


def test_get_pvalue(analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analyser = OLSAnalysis()
    assert analyser.get_pvalue(analysis_df_full) >= 0
