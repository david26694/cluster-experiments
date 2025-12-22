from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from cluster_experiments import (
    AnalysisPlan,
    LiftRegressionTransformer,
    OLSAnalysis,
    PowerAnalysis,
)


@pytest.fixture
def user_df(n_users=10_000):
    df = pd.DataFrame(
        {
            "customer_id": np.arange(n_users),
            "orders_pre": np.random.poisson(10, n_users),
            "treatment": np.random.rand(n_users) > 0.5,
            "X1": np.random.poisson(1, n_users),
            "X2": np.random.poisson(2, n_users),
        }
    )
    df = df.assign(
        **{
            "treatment": df["treatment"].astype(int),
            "orders": lambda x: x["orders_pre"]
            + 2 * x["X1"]
            + x["X2"]
            + x["treatment"],
        }
    )
    df = df.assign(
        **{
            "center_X1": lambda x: x["X1"] - x["X1"].mean(),
            "center_X2": lambda x: x["X2"] - x["X2"].mean(),
        }
    )
    return df


all_covariates_parametrize = pytest.mark.parametrize(
    "formula, covariates",
    [
        ("orders ~ treatment", []),
        ("orders ~ treatment + center_X1 + center_X2", ["center_X1", "center_X2"]),
        ("orders ~ treatment + X1 + X2", ["X1", "X2"]),
        ("orders ~ treatment + X1", ["X1"]),
    ],
)


@all_covariates_parametrize
def test_relative_lift_point_estimate(user_df, formula, covariates):
    # point estimates don't vary from naive to coavariates
    # given
    ols = sm.OLS.from_formula(formula, data=user_df).fit()
    control_mean = user_df.query("treatment == 0")["orders"].mean()

    # when
    transformer = LiftRegressionTransformer("treatment")
    transformer.fit(ols, user_df, covariates)

    # then
    assert transformer.params["treatment"] == pytest.approx(
        ols.params["treatment"] / control_mean, rel=1e-4
    )


@all_covariates_parametrize
def test_relative_lift_se(user_df, formula, covariates):
    # in the naive mode, we would underestimate control variance, but not by a lot
    # given
    ols = sm.OLS.from_formula(formula, data=user_df).fit()
    control_mean = user_df.query("treatment == 0")["orders"].mean()

    # when
    transformer = LiftRegressionTransformer("treatment")
    transformer.fit(ols, user_df, covariates)

    # then
    assert transformer.bse["treatment"] == pytest.approx(
        ols.bse["treatment"] / control_mean, rel=5e-2
    )
    assert transformer.bse["treatment"] >= ols.bse["treatment"] / control_mean


def test_relative_lift_covariates(user_df):
    # centering covariates shouldn't have an effect
    # given
    ols = sm.OLS.from_formula("orders ~ treatment + X1 + X2", data=user_df).fit()
    ols_centered = sm.OLS.from_formula(
        "orders ~ treatment + center_X1 + center_X2", data=user_df
    ).fit()

    # when
    transformer_ols = LiftRegressionTransformer("treatment")
    transformer_ols.fit(ols, user_df, ["X1", "X2"])

    transformer_centered = LiftRegressionTransformer("treatment")
    transformer_centered.fit(ols_centered, user_df, ["center_X1", "center_X2"])

    # then
    assert transformer_ols.bse["treatment"] == pytest.approx(
        transformer_centered.bse["treatment"], rel=1e-2
    )
    assert transformer_ols.params["treatment"] == pytest.approx(
        transformer_centered.params["treatment"], rel=1e-2
    )


@all_covariates_parametrize
def test_ci_and_pvalue(user_df, formula, covariates):
    # pvalue and confidence intervals are consistent
    # given
    ols = sm.OLS.from_formula(formula, data=user_df).fit()

    # when
    transformer = LiftRegressionTransformer("treatment")
    transformer.fit(ols, user_df, covariates)

    for alpha in [0.05, 0.01, 0.001]:
        ci = transformer.conf_int(alpha).loc["treatment"]
        # then
        if transformer.pvalues["treatment"] < alpha:
            assert ci[0] * ci[1] > 0
        else:
            assert ci[0] * ci[1] < 0


@all_covariates_parametrize
def test_mean_one(user_df, formula, covariates):
    # given control outcome has mean 1, it should be the same as vainilla effect
    user_df = user_df.copy()
    mean_control = user_df.query("treatment == 0")["orders"].mean()
    user_df["orders"] = user_df["orders"] / mean_control
    ols = sm.OLS.from_formula(formula, data=user_df).fit()

    # when
    transformer = LiftRegressionTransformer("treatment")
    transformer.fit(ols, user_df, covariates)

    # then
    assert transformer.params["treatment"] == pytest.approx(
        ols.params["treatment"], rel=1e-4
    )


@all_covariates_parametrize
def test_integration_experiment_analysis(user_df, formula, covariates):
    user_df = user_df.copy()
    user_df["treatment"] = user_df["treatment"].map({0: "B", 1: "A"})

    p_value = OLSAnalysis(
        target_col="orders", covariates=covariates, relative_effect=True
    ).get_pvalue(user_df)

    assert p_value < 0.05


def test_config_power_relative():
    # given
    config = {
        "analysis": "ols",
        "perturbator": "constant",
        "splitter": "non_clustered",
        "relative_effect": True,
    }

    # when
    pw = PowerAnalysis.from_dict(config)

    # then
    assert pw.analysis.relative_effect


@all_covariates_parametrize
def test_plan_config_relative(user_df, formula, covariates):
    # given
    user_df = user_df.copy()
    user_df["treatment"] = user_df["treatment"].map({0: "A", 1: "B"})
    config = {
        "metrics": [
            {"alias": "Orders", "name": "orders"},
        ],
        "variants": [
            {"name": "A", "is_control": True},
            {"name": "B", "is_control": False},
        ],
        "analysis_type": "ols",
        "variant_col": "treatment",
        "analysis_config": {"relative_effect": True, "covariates": covariates},
    }
    non_relative_config = deepcopy(config)
    non_relative_config["analysis_config"] = {"covariates": covariates}
    control_mean = user_df.query("treatment == 'A'")["orders"].mean()

    # when
    plan = AnalysisPlan.from_metrics_dict(config)
    non_relative_plan = AnalysisPlan.from_metrics_dict(non_relative_config)
    results = plan.analyze(user_df)
    results_abs = non_relative_plan.analyze(user_df)

    # then
    assert plan.tests[0].experiment_analysis.relative_effect
    assert results.ate[0] == pytest.approx(results_abs.ate[0] / control_mean, rel=1e-4)
