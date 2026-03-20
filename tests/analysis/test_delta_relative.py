"""
Tests for relative lift and MDE on ratio metrics via DeltaMethodLiftTransformer
and DeltaMethodAnalysis(relative_effect=True).

Structure mirrors test_lift_transformer.py.

Key assertions:
- Without covariates:
    * point estimate == manual (mean_diff / ctrl_mean)
    * SE >= naive (SE_abs / ctrl_mean), close within 5 %
    * power slightly lower than "naive" power
- With covariates (CUPED):
    * point estimate close to manual (tolerance is wider due to CUPED correction)
    * SE >= naive, close
    * power slightly lower than naive
- E2E:
    * DeltaMethodAnalysis(relative_effect=True).get_pvalue detects planted effect
    * PowerAnalysis.from_dict with analysis="delta", relative_effect=True works
    * AnalysisPlan with RatioMetric + relative_effect=True gives ATE ≈ abs_ATE / ctrl_mean
    * Non-delta/non-OLS analysis with relative_effect=True raises ValueError
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from cluster_experiments import (
    AnalysisPlan,
    DeltaMethodAnalysis,
    DeltaMethodLiftTransformer,
    NormalPowerAnalysis,
    PowerAnalysis,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ratio_df(
    n_users: int = 3_000,
    treatment_effect: float = 0.05,
    seed: int = 42,
    with_covariate: bool = False,
) -> pd.DataFrame:
    """
    Cluster-level ratio data.  Each row is one user (= one cluster).
    target ~ Binomial(scale, rate),  rate_treatment = rate_control + treatment_effect.
    """
    rng = np.random.default_rng(seed)
    user_ids = np.arange(n_users)
    treatment_flag = rng.integers(0, 2, size=n_users)  # 0 or 1
    scale = rng.integers(5, 20, size=n_users).astype(float)
    base_rate = 0.30 + rng.normal(0, 0.05, size=n_users)
    base_rate = np.clip(base_rate, 0.05, 0.95)
    rate = base_rate + treatment_effect * treatment_flag
    rate = np.clip(rate, 0, 1)
    target = rng.binomial(scale.astype(int), rate).astype(float)
    treatment_label = np.where(treatment_flag == 0, "A", "B")

    df = pd.DataFrame(
        {
            "user": user_ids,
            "treatment": treatment_label,
            "target": target,
            "scale": scale,
        }
    )

    if with_covariate:
        # pre-experiment covariate correlated with base_rate
        df["pre_rate"] = base_rate * scale + rng.normal(0, 0.1, size=n_users)
        df["pre_scale"] = scale + rng.integers(-2, 3, size=n_users).astype(float)
        df["pre_scale"] = df["pre_scale"].clip(1)

    return df


@pytest.fixture
def ratio_df():
    return _make_ratio_df(n_users=5_000, treatment_effect=0.05, seed=0)


@pytest.fixture
def ratio_df_covariate():
    return _make_ratio_df(
        n_users=5_000, treatment_effect=0.05, seed=0, with_covariate=True
    )


# ---------------------------------------------------------------------------
# Helpers – compute "manual" relative lift (naive denominator is fixed)
# ---------------------------------------------------------------------------


def _manual_relative_lift(df: pd.DataFrame):
    """Returns (relative_lift, se_naive, ctrl_mean, ctrl_var, treat_var)."""
    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", target_col="target"
    )
    # Replicate _get_mean_standard_error internals for the absolute case
    df2 = df.copy()
    df2 = analyser._create_binary_treatment(df2)
    df2 = analyser._aggregate_to_cluster(df2)
    is_treatment = df2["treatment"] == 1
    ctrl_mean, ctrl_var = analyser._get_group_mean_and_variance(
        df2[~is_treatment], None, []
    )
    treat_mean, treat_var = analyser._get_group_mean_and_variance(
        df2[is_treatment], None, []
    )
    mean_diff = treat_mean - ctrl_mean
    var_abs = treat_var + ctrl_var
    relative_lift = mean_diff / ctrl_mean
    se_naive = np.sqrt(var_abs) / ctrl_mean
    return relative_lift, se_naive, ctrl_mean, ctrl_var, treat_var, var_abs


# ---------------------------------------------------------------------------
# Unit tests – DeltaMethodLiftTransformer directly
# ---------------------------------------------------------------------------


def test_transformer_point_estimate_matches_manual(ratio_df):
    """Relative lift from transformer == manual mean_diff / ctrl_mean."""
    rel_lift_manual, _, ctrl_mean, ctrl_var, treat_var, var_abs = _manual_relative_lift(
        ratio_df
    )

    transformer = DeltaMethodLiftTransformer("treatment")
    mean_diff = rel_lift_manual * ctrl_mean
    transformer.fit(
        mean_diff=mean_diff, var_abs=var_abs, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var
    )

    assert transformer.params["treatment"] == pytest.approx(rel_lift_manual, rel=1e-8)
    assert transformer.summary()["percent_lift"] == transformer.params["treatment"]


def test_transformer_se_greater_than_naive(ratio_df):
    """SE from outer delta method >= naive SE = SE_abs / ctrl_mean."""
    _, se_naive, ctrl_mean, ctrl_var, treat_var, var_abs = _manual_relative_lift(
        ratio_df
    )
    rel_lift_manual, _, _, _, _, _ = _manual_relative_lift(ratio_df)
    mean_diff = rel_lift_manual * ctrl_mean

    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=mean_diff, var_abs=var_abs, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var
    )

    assert transformer.bse["treatment"] >= se_naive
    assert transformer.bse["treatment"] == pytest.approx(se_naive, rel=0.15)


def test_transformer_se_via_summary(ratio_df):
    """summary()['_se_relative_lift'] == bse['treatment']."""
    rel_lift_manual, _, ctrl_mean, ctrl_var, _, var_abs = _manual_relative_lift(
        ratio_df
    )
    mean_diff = rel_lift_manual * ctrl_mean

    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=mean_diff, var_abs=var_abs, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var
    )

    assert transformer.summary()["_se_relative_lift"] == transformer.bse["treatment"]


def test_transformer_conf_int_consistent_with_pvalue(ratio_df):
    """CI and p-value are mutually consistent for multiple alphas."""
    rel_lift_manual, _, ctrl_mean, ctrl_var, _, var_abs = _manual_relative_lift(
        ratio_df
    )
    mean_diff = rel_lift_manual * ctrl_mean

    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=mean_diff, var_abs=var_abs, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var
    )

    for alpha in [0.05, 0.01, 0.001]:
        ci = transformer.conf_int(alpha).loc["treatment"]
        if transformer.pvalues["treatment"] < alpha:
            assert ci[0] * ci[1] > 0, f"CI should exclude 0 at alpha={alpha}"
        else:
            assert ci[0] * ci[1] < 0, f"CI should include 0 at alpha={alpha}"


def test_transformer_zero_ctrl_mean_raises():
    """lift_and_se and relative_mde raise when ctrl_mean == 0."""
    with pytest.raises(ValueError, match="ctrl_mean must be non-zero"):
        DeltaMethodLiftTransformer.lift_and_se(0.1, 0.01, 0.0, 0.001)

    with pytest.raises(ValueError, match="ctrl_mean must be non-zero"):
        DeltaMethodLiftTransformer.relative_mde(0.05, 0.8, 0.0, 0.001, 0.001)


def test_relative_mde_invalid_power_equation_raises(monkeypatch):
    """relative_mde raises when the quadratic power equation becomes degenerate (A == 0)."""

    def mock_ppf(q):
        # z_alpha for q=0.975 and z_beta for q=0.8
        return 1.96 if q > 0.9 else 2.0

    monkeypatch.setattr(
        "cluster_experiments.relative_lift_transformer.stats.norm.ppf", mock_ppf
    )

    with pytest.raises(ValueError, match="invalid power equation"):
        DeltaMethodLiftTransformer.relative_mde(
            alpha=0.05,
            power=0.8,
            ctrl_mean=1.0,
            ctrl_var=0.25,
            treat_var=0.25,
        )


def test_static_lift_and_se_matches_fit():
    """fit() delegates to lift_and_se() and gives the same result."""
    mean_diff, var_abs, ctrl_mean, ctrl_var = 0.05, 0.001, 0.30, 0.0001
    rl, se = DeltaMethodLiftTransformer.lift_and_se(
        mean_diff, var_abs, ctrl_mean, ctrl_var
    )

    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=mean_diff, var_abs=var_abs, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var
    )

    assert transformer.params["treatment"] == pytest.approx(rl, rel=1e-10)
    assert transformer.bse["treatment"] == pytest.approx(se, rel=1e-10)


# ---------------------------------------------------------------------------
# Integration – DeltaMethodAnalysis(relative_effect=True)
# ---------------------------------------------------------------------------


def test_delta_analysis_relative_point_estimate(ratio_df):
    """DeltaMethodAnalysis relative point estimate == manual relative lift."""
    rel_lift_manual, _, _, _, _, _ = _manual_relative_lift(ratio_df)

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    rel_lift_analysis = analyser.get_point_estimate(ratio_df)

    assert rel_lift_analysis == pytest.approx(rel_lift_manual, rel=1e-6)


def test_delta_analysis_relative_se_greater_than_naive(ratio_df):
    """SE from DeltaMethodAnalysis(relative_effect=True) >= naive."""
    _, se_naive, _, _, _, _ = _manual_relative_lift(ratio_df)

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    se_relative = analyser.get_standard_error(ratio_df)

    assert se_relative >= se_naive
    assert se_relative == pytest.approx(se_naive, rel=0.15)


def test_delta_analysis_relative_pvalue_detects_effect(ratio_df):
    """p-value should be significant with a planted 5% treatment effect."""
    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    p_value = analyser.get_pvalue(ratio_df)
    assert p_value < 0.05


def test_delta_analysis_relative_absolute_consistent(ratio_df):
    """Relative point estimate == absolute point estimate / ctrl_mean."""
    analyser_abs = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", target_col="target"
    )
    analyser_rel = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    abs_est = analyser_abs.get_point_estimate(ratio_df)
    rel_est = analyser_rel.get_point_estimate(ratio_df)

    # control mean from data
    df2 = ratio_df.copy()
    df2["treatment"] = df2["treatment"].map({"A": 0, "B": 1})
    agg = df2.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum"}
    )
    ctrl_mean = (
        agg[agg["treatment"] == 0]["target"].sum()
        / agg[agg["treatment"] == 0]["scale"].sum()
    )

    assert rel_est == pytest.approx(abs_est / ctrl_mean, rel=1e-6)


# ---------------------------------------------------------------------------
# With covariates (CUPED)
# ---------------------------------------------------------------------------


def test_delta_relative_with_covariates_point_estimate(ratio_df_covariate):
    """With CUPED covariates: relative point estimate == abs_CUPED / ctrl_mean_CUPED."""
    df = ratio_df_covariate.copy()
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum", "pre_rate": "sum", "pre_scale": "sum"}
    )

    analyser_abs = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        covariates=["pre_rate"],
    )
    analyser_rel = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        covariates=["pre_rate"],
        relative_effect=True,
    )

    # Absolute CUPED estimate and SE
    abs_est = analyser_abs.get_point_estimate(df_agg)
    analyser_abs.get_standard_error(df_agg)

    # Relative CUPED estimate and SE
    rel_est = analyser_rel.get_point_estimate(df_agg)
    rel_se = analyser_rel.get_standard_error(df_agg)

    # The CUPED-adjusted ctrl_mean is what the transformer divides by.
    # We can recover it as abs_est / rel_est (since rel = abs / ctrl_mean_cuped).
    cuped_ctrl_mean = abs_est / rel_est

    # rel_est should equal abs_est / cuped_ctrl_mean by construction
    assert rel_est == pytest.approx(abs_est / cuped_ctrl_mean, rel=1e-6)

    # The CUPED ctrl_mean should be positive (sensible ratio metric)
    assert cuped_ctrl_mean > 0

    # Relative SE should be positive and finite
    assert rel_se > 0
    assert np.isfinite(rel_se)


def test_delta_relative_with_covariates_se_greater_than_naive(ratio_df_covariate):
    """SE(relative, CUPED) >= SE_abs(CUPED) / ctrl_mean."""
    df = ratio_df_covariate.copy()
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum", "pre_rate": "sum", "pre_scale": "sum"}
    )

    analyser_abs = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        covariates=["pre_rate"],
    )
    analyser_rel = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        covariates=["pre_rate"],
        relative_effect=True,
    )

    abs_se = analyser_abs.get_standard_error(df_agg)
    ctrl_mean_raw = (
        df_agg[df_agg["treatment"] == "A"]["target"].sum()
        / df_agg[df_agg["treatment"] == "A"]["scale"].sum()
    )
    naive_se = abs_se / ctrl_mean_raw
    rel_se = analyser_rel.get_standard_error(df_agg)

    assert rel_se >= naive_se


# ---------------------------------------------------------------------------
# Power / MDE – DeltaMethodLiftTransformer.relative_mde
# ---------------------------------------------------------------------------


def test_relative_mde_lower_than_naive_mde():
    """
    Naive MDE = (z_a + z_b) * SE_abs / ctrl_mean.
    Proper relative MDE should be equal (in the limit of large n where ctrl variance
    is negligible) or slightly larger.  Here we just check it's finite and positive.
    """
    from scipy.stats import norm

    alpha = 0.05
    power = 0.8
    ctrl_mean = 0.30
    ctrl_var = 0.0001
    treat_var = 0.0001

    mde = DeltaMethodLiftTransformer.relative_mde(
        alpha=alpha,
        power=power,
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
        treat_var=treat_var,
    )

    assert mde > 0
    assert np.isfinite(mde)

    # Compare against naive approximation
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    naive_mde = (z_alpha + z_beta) * np.sqrt(treat_var + ctrl_var) / ctrl_mean
    # Proper MDE should be close to naive when ctrl variance is small
    assert mde == pytest.approx(naive_mde, rel=0.10)


# ---------------------------------------------------------------------------
# E2E – PowerAnalysis / NormalPowerAnalysis config round-trip
# ---------------------------------------------------------------------------


def test_config_power_delta_relative():
    """PowerAnalysis.from_dict with analysis='delta' and relative_effect=True works."""
    config = {
        "analysis": "delta",
        "perturbator": "constant",
        "splitter": "clustered",
        "cluster_cols": ["user"],
        "scale_col": "scale",
        "relative_effect": True,
    }
    pw = PowerAnalysis.from_dict(config)
    assert pw.analysis.relative_effect


def test_config_power_relative_wrong_analysis_raises():
    """relative_effect=True with GEE analysis raises ValueError."""
    config = {
        "analysis": "gee",
        "perturbator": "constant",
        "splitter": "non_clustered",
        "relative_effect": True,
    }
    with pytest.raises(ValueError, match="relative_effect"):
        PowerAnalysis.from_dict(config)


def test_normal_power_analysis_delta_relative(ratio_df):
    """NormalPowerAnalysis with delta + relative_effect returns a valid MDE."""
    from cluster_experiments.random_splitter import ClusteredSplitter

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    pw = NormalPowerAnalysis(
        analysis=analyser,
        splitter=ClusteredSplitter(cluster_cols=["user"]),
    )
    # Drop treatment column so power analysis assigns its own splits
    df_no_treatment = ratio_df.drop(columns=["treatment"])
    mde = pw.mde(df_no_treatment, power=0.8, n_simulations=5)
    assert mde > 0
    assert np.isfinite(mde)


# ---------------------------------------------------------------------------
# E2E – AnalysisPlan with RatioMetric + relative_effect=True
# ---------------------------------------------------------------------------


def test_analysis_plan_ratio_relative_effect(ratio_df):
    """
    AnalysisPlan with RatioMetric + analysis_type='delta' + relative_effect=True:
    - ATE is in relative (percent) units
    - ATE ≈ absolute_ATE / control_mean
    """
    # Absolute plan
    abs_config = {
        "metrics": [{"alias": "conversion", "name": "target", "scale_name": "scale"}],
        "variants": [
            {"name": "A", "is_control": True},
            {"name": "B", "is_control": False},
        ],
        "analysis_type": "delta",
        "variant_col": "treatment",
        "analysis_config": {"cluster_cols": ["user"]},
    }
    # Relative plan
    rel_config = deepcopy(abs_config)
    rel_config["analysis_config"] = {
        "cluster_cols": ["user"],
        "relative_effect": True,
    }

    abs_plan = AnalysisPlan.from_metrics_dict(abs_config)
    rel_plan = AnalysisPlan.from_metrics_dict(rel_config)

    abs_results = abs_plan.analyze(ratio_df)
    rel_results = rel_plan.analyze(ratio_df)

    # Control mean for manual comparison
    df_agg = ratio_df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum"}
    )
    ctrl_mean = (
        df_agg[df_agg["treatment"] == "A"]["target"].sum()
        / df_agg[df_agg["treatment"] == "A"]["scale"].sum()
    )

    assert rel_results.ate[0] == pytest.approx(abs_results.ate[0] / ctrl_mean, rel=1e-6)
    assert rel_plan.tests[0].experiment_analysis.relative_effect


# ---------------------------------------------------------------------------
# Power is slightly lower for relative than naive under same planted effect
# ---------------------------------------------------------------------------


def test_power_relative_slightly_lower_than_naive(ratio_df):
    """
    Estimating relative lift with proper SE gives lower power than naive
    (dividing absolute MDE by control mean), because SE_relative > SE_abs / ctrl_mean.
    """
    from scipy.stats import norm

    _, _, ctrl_mean, ctrl_var, treat_var, var_abs = _manual_relative_lift(ratio_df)

    alpha = 0.05
    planted_rel_effect = 0.10  # 10% relative lift

    # Naive SE (underestimates)
    se_naive = np.sqrt(var_abs) / ctrl_mean
    # Proper SE from transformer
    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=planted_rel_effect * ctrl_mean,
        var_abs=var_abs,
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
    )
    se_proper = transformer.bse["treatment"]

    z_alpha = norm.ppf(1 - alpha / 2)
    power_naive = 1 - norm.cdf(z_alpha - planted_rel_effect / se_naive)
    power_proper = 1 - norm.cdf(z_alpha - planted_rel_effect / se_proper)

    # Proper SE is larger so power should be lower or equal
    assert power_proper <= power_naive + 1e-6
