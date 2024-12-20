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


def test_cov_type(analysis_df):
    # given
    analyser_hc1 = OLSAnalysis(cov_type="HC1")
    analyser_hc3 = OLSAnalysis(cov_type="HC3")

    # then: point estimates are the same
    assert analyser_hc1.get_point_estimate(
        analysis_df
    ) == analyser_hc3.get_point_estimate(analysis_df)

    # then: standard errors are different
    assert analyser_hc1.get_standard_error(
        analysis_df
    ) != analyser_hc3.get_standard_error(analysis_df)
