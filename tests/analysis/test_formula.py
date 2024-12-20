import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    GeeExperimentAnalysis,
    MLMExperimentAnalysis,
)

parametrisation = pytest.mark.parametrize(
    "analysis_class",
    [
        ClusteredOLSAnalysis,
        GeeExperimentAnalysis,
        MLMExperimentAnalysis,
    ],
)


@parametrisation
def test_formula_no_covariates(analysis_class):
    analysis = analysis_class(
        cluster_cols=["cluster"],
        treatment_col="treatment",
        target_col="y",
    )

    assert analysis.formula == "y ~ treatment"


@parametrisation
def test_formula_with_covariates(analysis_class):
    analysis = analysis_class(
        cluster_cols=["cluster"],
        treatment_col="treatment",
        target_col="y",
        covariates=["covariate1", "covariate2"],
    )

    assert analysis.formula == "y ~ treatment + covariate1 + covariate2"


@parametrisation
def test_formula_with_interaction(analysis_class):
    analysis = analysis_class(
        cluster_cols=["cluster"],
        treatment_col="treatment",
        target_col="y",
        covariates=["covariate1", "covariate2"],
        add_covariate_interaction=True,
    )

    assert (
        analysis.formula
        == "y ~ treatment + covariate1 + covariate2 + __covariate1__interaction + __covariate2__interaction"
    )
