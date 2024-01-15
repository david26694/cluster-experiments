import pandas as pd
import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    GeeExperimentAnalysis,
    MLMExperimentAnalysis,
    OLSAnalysis,
    TTestClusteredAnalysis,
)
from tests.utils import generate_clustered_data


@pytest.mark.parametrize("hypothesis", ["less", "greater", "two-sided"])
@pytest.mark.parametrize("analysis_class", [OLSAnalysis])
def test_get_pvalue_hypothesis(analysis_class, hypothesis, analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analyser = analysis_class(hypothesis=hypothesis)
    assert analyser.get_pvalue(analysis_df_full) >= 0


@pytest.mark.parametrize("hypothesis", ["less", "greater", "two-sided"])
@pytest.mark.parametrize(
    "analysis_class",
    [
        ClusteredOLSAnalysis,
        GeeExperimentAnalysis,
        TTestClusteredAnalysis,
        MLMExperimentAnalysis,
    ],
)
def test_get_pvalue_hypothesis_clustered(analysis_class, hypothesis):

    analysis_df_full = generate_clustered_data()
    analyser = analysis_class(hypothesis=hypothesis, cluster_cols=["user_id"])
    assert analyser.get_pvalue(analysis_df_full) >= 0


@pytest.mark.parametrize("analysis_class", [OLSAnalysis])
def test_get_pvalue_hypothesis_default(analysis_class, analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analyser = analysis_class()
    assert analyser.get_pvalue(analysis_df_full) >= 0


@pytest.mark.parametrize("analysis_class", [OLSAnalysis])
def test_get_pvalue_hypothesis_wrong_input(analysis_class, analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])

    # Use pytest.raises to check for ValueError
    with pytest.raises(ValueError) as excinfo:
        analyser = analysis_class(hypothesis="wrong_input")
        analyser.get_pvalue(analysis_df_full) >= 0

    # Check if the error message is as expected
    assert "'wrong_input' is not a valid HypothesisEntries" in str(excinfo.value)


@pytest.mark.parametrize("analysis_class", [OLSAnalysis])
def test_several_hypothesis(analysis_class, analysis_df):
    analysis_df_full = pd.concat([analysis_df for _ in range(100)])
    analysis_less = analysis_class(hypothesis="less")
    analysis_greater = analysis_class(hypothesis="greater")
    analysis_two_sided = analysis_class(hypothesis="two-sided")

    assert (
        analysis_less.get_pvalue(analysis_df_full)
        == analysis_two_sided.get_pvalue(analysis_df_full) / 2
    )
    assert (
        analysis_greater.get_pvalue(analysis_df_full)
        == 1 - analysis_two_sided.get_pvalue(analysis_df_full) / 2
    )
