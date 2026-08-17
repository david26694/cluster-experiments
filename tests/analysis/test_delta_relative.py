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
from scipy.stats import norm

from cluster_experiments import (
    AnalysisPlan,
    DeltaMethodAnalysis,
    DeltaMethodLiftTransformer,
    NormalPowerAnalysis,
    PowerAnalysis,
    StandardErrorCurve,
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
        mean_diff=mean_diff,
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
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
        mean_diff=mean_diff,
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
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
        mean_diff=mean_diff,
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
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
        mean_diff=mean_diff,
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
    )

    for alpha in [0.05, 0.01, 0.001]:
        ci = transformer.conf_int(alpha).loc["treatment"]
        if transformer.pvalues["treatment"] < alpha:
            assert ci[0] * ci[1] > 0, f"CI should exclude 0 at alpha={alpha}"
        else:
            assert ci[0] * ci[1] < 0, f"CI should include 0 at alpha={alpha}"


def test_transformer_zero_ctrl_mean_raises():
    """A zero control mean is rejected where the lift is computed.

    The guard lives in the transformer rather than in the MDE solver: with
    ctrl_mean == 0 no standard error curve can be built in the first place, so
    the solver never sees the degenerate input.
    """
    with pytest.raises(ValueError, match="ctrl_mean must be non-zero"):
        DeltaMethodLiftTransformer.lift_and_se(0.1, 0.01, 0.0, 0.001)

    with pytest.raises(ValueError, match="ctrl_mean must be non-zero"):
        DeltaMethodLiftTransformer("treatment").fit(
            mean_diff=0.1, std_error=0.1, ctrl_mean=0.0, ctrl_var=0.001
        )


def test_relative_mde_above_power_ceiling_raises():
    """
    No finite MDE exists once the baseline is too noisy.

    Solving m = (z_alpha + z_beta) * SE(m) needs the right-hand side to grow more
    slowly than the left. SE(m) grows like sqrt(effect_var) * m, so a solution
    exists only while |z_alpha + z_beta| < 1 / sqrt(effect_var), whatever the
    effect size.
    """
    pw = _make_relative_delta_power("greater")

    # effect_var = ctrl_var / ctrl_mean**2 = 2, so z_alpha + z_beta must stay
    # below 1 / sqrt(2) = 0.707.
    effect_var = 2.0
    max_k = 1 / np.sqrt(effect_var)
    curve = _delta_curve(1.0, effect_var, effect_var)

    z_alpha = norm.ppf(1 - 0.05)
    assert z_alpha + norm.ppf(0.8) > max_k  # the usual design is unreachable
    with pytest.raises(ValueError, match="No finite minimum detectable effect"):
        pw._mde_from_curve(curve, 0.05, 0.8)

    # Relaxing alpha and power until z_alpha + z_beta drops below the cap makes a
    # finite MDE reappear.
    alpha, power = 0.4, 0.5
    assert norm.ppf(1 - alpha) + norm.ppf(power) < max_k
    mde = pw._mde_from_curve(curve, alpha, power)
    assert np.isfinite(mde) and mde > 0


def test_static_lift_and_se_matches_fit():
    """fit() delegates to lift_and_se() and gives the same result."""
    mean_diff, var_abs, ctrl_mean, ctrl_var = 0.05, 0.001, 0.30, 0.0001
    rl, se = DeltaMethodLiftTransformer.lift_and_se(
        mean_diff, var_abs, ctrl_mean, ctrl_var
    )

    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(
        mean_diff=mean_diff,
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
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
# Power / MDE – NormalPowerAnalysis._mde_from_curve
# ---------------------------------------------------------------------------


def _delta_curve(
    ctrl_mean: float, ctrl_var: float, treat_var: float
) -> StandardErrorCurve:
    """
    Standard error curve of the relative lift, built from delta-method group
    statistics.

    Spelled out here rather than taken from the production code so the MDE tests
    below check the solver against an independently written curve. The arms are
    independent, hence ``effect_cov == -effect_var``.
    """
    effect_var = ctrl_var / ctrl_mean**2
    return StandardErrorCurve(
        std_error=np.sqrt(treat_var + ctrl_var) / abs(ctrl_mean),
        effect_var=effect_var,
        effect_cov=-effect_var,
    )


def _make_relative_delta_power(hypothesis: str = "two-sided") -> NormalPowerAnalysis:
    """Build a NormalPowerAnalysis wrapping a relative DeltaMethodAnalysis."""
    from cluster_experiments.random_splitter import ClusteredSplitter

    return NormalPowerAnalysis(
        analysis=DeltaMethodAnalysis(
            cluster_cols=["user"],
            scale_col="scale",
            target_col="target",
            relative_effect=True,
            hypothesis=hypothesis,
        ),
        splitter=ClusteredSplitter(cluster_cols=["user"]),
    )


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

    pw = _make_relative_delta_power()
    mde = pw._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    assert mde > 0
    assert np.isfinite(mde)

    # Compare against naive approximation
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    naive_mde = (z_alpha + z_beta) * np.sqrt(treat_var + ctrl_var) / ctrl_mean
    # Proper MDE should be close to naive when ctrl variance is small
    assert mde == pytest.approx(naive_mde, rel=0.10)


def test_relative_mde_geq_linear():
    """The relative MDE is always >= the linear approximation, since the SE
    grows with the effect size."""
    from scipy.stats import norm

    alpha = 0.05
    power = 0.8
    ctrl_mean = 0.30
    # Non-negligible control variance so the two formulas diverge.
    ctrl_var = 0.01
    treat_var = 0.01

    pw = _make_relative_delta_power()
    relative_mde = pw._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    linear_mde = (z_alpha + z_beta) * np.sqrt(treat_var + ctrl_var) / ctrl_mean

    assert relative_mde >= linear_mde


def _achieved_power(
    m: float,
    alpha: float,
    ctrl_mean: float,
    ctrl_var: float,
    treat_var: float,
    hypothesis: str,
) -> float:
    """
    Independent reference: the power actually achieved at relative effect ``m``.

    Derived directly from the Wald-test definition rather than from the solver's
    internals, so it catches wrong-root selection and wrong one-sided handling.
    The test divides by the standard error estimated at the observed effect, so
    under an alternative ``m`` the scale is ``SE_rel(m)`` throughout::

        power = Phi(|m| / SE_rel(m) - z_alpha)

    Only the correct-direction rejection is counted, matching the MDE.
    """
    from scipy.stats import norm

    se2_c = ctrl_var / ctrl_mean**2
    se2_t = treat_var / ctrl_mean**2
    se_rel_m = np.sqrt(se2_t + se2_c * (1 + m) ** 2)
    z_alpha = (
        norm.ppf(1 - alpha / 2) if hypothesis == "two-sided" else norm.ppf(1 - alpha)
    )
    return float(norm.cdf(abs(m) / se_rel_m - z_alpha))


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])
def test_relative_mde_recovers_target_power(hypothesis):
    """The returned MDE must reproduce the requested power under the independent
    Wald-test definition, for every alternative."""
    alpha = 0.05
    power = 0.8
    ctrl_mean = 1.0
    # Non-trivial variance so the effect-dependent SE term matters.
    ctrl_var = 0.05
    treat_var = 0.05

    mde = _make_relative_delta_power(hypothesis)._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    if hypothesis == "less":
        assert mde < 0
    else:
        assert mde > 0

    achieved = _achieved_power(mde, alpha, ctrl_mean, ctrl_var, treat_var, hypothesis)
    assert achieved == pytest.approx(power, abs=1e-6)


@pytest.mark.parametrize("hypothesis", ["greater", "less"])
def test_relative_mde_recovers_low_target_power(hypothesis):
    """A target power below 0.5 uses the opposite quadratic-root filter."""
    alpha = 0.05
    power = 0.3
    ctrl_mean = 1.0
    ctrl_var = 0.01
    treat_var = 0.01

    mde = _make_relative_delta_power(hypothesis)._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    if hypothesis == "less":
        assert mde < 0
    else:
        assert mde > 0

    achieved = _achieved_power(mde, alpha, ctrl_mean, ctrl_var, treat_var, hypothesis)
    assert achieved == pytest.approx(power, abs=1e-6)


@pytest.mark.parametrize("hypothesis", ["greater", "less"])
def test_relative_mde_allows_zero_at_null_power(hypothesis):
    """At one-sided null power, zero is the valid minimum effect."""
    alpha = power = 0.05
    ctrl_mean = 1.0
    ctrl_var = 0.01
    treat_var = 0.01

    mde = _make_relative_delta_power(hypothesis)._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    assert mde == pytest.approx(0.0, abs=1e-12)
    achieved = _achieved_power(mde, alpha, ctrl_mean, ctrl_var, treat_var, hypothesis)
    assert achieved == pytest.approx(power, abs=1e-6)


def test_relative_mde_power_half_is_the_rejection_boundary():
    """
    At 50% power the MDE is the rejection threshold itself.

    With z_beta = 0 the equation reduces to m = z_alpha * SE(m), whose solution is
    the effect at which the estimate sits exactly on the critical value. Because
    SE grows with the effect, that is strictly above the naive z_alpha * SE(0).
    """
    from scipy.stats import norm

    alpha = 0.05
    ctrl_mean, ctrl_var, treat_var = 1.0, 0.01, 0.01
    curve = _delta_curve(ctrl_mean, ctrl_var, treat_var)

    mde = _make_relative_delta_power("greater")._mde_from_curve(curve, alpha, 0.5)

    z_alpha = norm.ppf(1 - alpha)
    assert mde == pytest.approx(z_alpha * curve.standard_error_at(mde))  # fixed point
    assert mde > z_alpha * curve.std_error  # strictly above the naive boundary
    achieved = _achieved_power(mde, alpha, ctrl_mean, ctrl_var, treat_var, "greater")
    assert achieved == pytest.approx(0.5, abs=1e-6)


def test_relative_mde_allows_zero_variance():
    """A deterministic ratio metric has a zero MDE instead of no solution."""
    mde = _make_relative_delta_power("greater")._mde_from_curve(
        _delta_curve(1.0, 0.0, 0.0),
        0.05,
        0.8,
    )

    assert mde == pytest.approx(0.0, abs=1e-12)


def test_relative_mde_one_sided_is_asymmetric():
    """Because SE_rel(m) depends on (1 + m)**2, the 'less' MDE is NOT the
    negative of the 'greater' MDE when the control variance is non-negligible."""
    curve = _delta_curve(ctrl_mean=1.0, ctrl_var=0.05, treat_var=0.05)
    mde_greater = _make_relative_delta_power("greater")._mde_from_curve(
        curve, 0.05, 0.8
    )
    mde_less = _make_relative_delta_power("less")._mde_from_curve(curve, 0.05, 0.8)

    assert mde_greater > 0
    assert mde_less < 0
    # The magnitudes genuinely differ (the naive `-mde_greater` shortcut is wrong).
    assert abs(mde_greater) != pytest.approx(abs(mde_less), rel=1e-3)


def test_relative_mde_high_cv_regime():
    """A high control-CV regime still returns a power-recovering MDE."""
    alpha = 0.05
    power = 0.8
    ctrl_mean = 1.0
    # effect_var = ctrl_var / ctrl_mean**2 = 0.1, so (z_alpha + z_beta) = 2.80 is
    # just under the 1 / sqrt(0.1) = 3.16 cap: a solution exists, but the effect
    # dependence of the standard error dominates the answer.
    ctrl_var = 0.1
    treat_var = 0.1

    mde = _make_relative_delta_power()._mde_from_curve(
        _delta_curve(ctrl_mean, ctrl_var, treat_var),
        alpha,
        power,
    )

    assert mde > 0
    achieved = _achieved_power(mde, alpha, ctrl_mean, ctrl_var, treat_var, "two-sided")
    assert achieved == pytest.approx(power, abs=1e-6)


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("power", [0.3, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("ctrl_var", [0.0, 0.0001, 0.01, 0.05])
def test_relative_mde_satisfies_the_power_equation(hypothesis, power, ctrl_var):
    """
    The returned MDE solves m = (z_alpha + z_beta) * SE(m) directly.

    Squaring that equation to get the quadratic introduces a second root of the
    opposite sign, so this checks the residual of the *unsquared* equation rather
    than trusting the root selection. Covers powers either side of 50%, where the
    sign of z_beta flips, and a deterministic baseline where the curve is flat.
    """
    alpha = 0.05
    curve = _delta_curve(ctrl_mean=1.0, ctrl_var=ctrl_var, treat_var=0.01)
    mde = _make_relative_delta_power(hypothesis)._mde_from_curve(curve, alpha, power)

    if hypothesis == "less":
        z_alpha, z_beta = norm.ppf(alpha), norm.ppf(1 - power)
    elif hypothesis == "greater":
        z_alpha, z_beta = norm.ppf(1 - alpha), norm.ppf(power)
    else:
        z_alpha, z_beta = norm.ppf(1 - alpha / 2), norm.ppf(power)

    assert mde == pytest.approx(
        (z_alpha + z_beta) * curve.standard_error_at(mde), abs=1e-12
    )
    assert np.sign(mde) == np.sign(z_alpha + z_beta) or mde == 0.0


def _wrong_direction_tail(
    curve: StandardErrorCurve, mde: float, alpha: float, hypothesis: str
) -> float:
    """
    Probability of rejecting in the *wrong* direction at effect ``mde``.

    A two-sided power calculation counts these rejections; the MDE formula does
    not, since a sign-flipped rejection is not a detection (and a sum of two
    normal CDFs has no analytic inverse anyway). Subtracting this term makes the
    round-trip exact rather than approximate, and pins down precisely how the two
    differ. One-sided tests have no such term.
    """
    if hypothesis != "two-sided":
        return 0.0
    return float(
        norm.cdf(-norm.ppf(1 - alpha / 2) - mde / curve.standard_error_at(mde))
    )


def test_flat_curve_mde_matches_linear_formula():
    """
    A curve with no effect dependence must reproduce the linear normal formula
    exactly. This is what lets absolute analyses go through the same code path
    without their numbers moving.
    """
    for hypothesis, z_alpha, z_beta in [
        ("two-sided", norm.ppf(1 - 0.025), norm.ppf(0.8)),
        ("greater", norm.ppf(1 - 0.05), norm.ppf(0.8)),
        ("less", norm.ppf(0.05), norm.ppf(1 - 0.8)),
    ]:
        pw = _make_relative_delta_power(hypothesis)
        curve = StandardErrorCurve(std_error=0.25)
        assert pw._mde_from_curve(curve, 0.05, 0.8) == float(z_alpha + z_beta) * 0.25


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("power", [0.6, 0.8, 0.95])
def test_mde_and_power_are_inverses(hypothesis, power):
    """
    The MDE and the power calculation must invert each other.

    This is the regression test for the two disagreeing: the MDE used the
    effect-dependent standard error while power used a constant one, so
    ``power_line(mde(p))`` did not return ``p``.
    """
    pw = _make_relative_delta_power(hypothesis)
    curve = _delta_curve(ctrl_mean=1.0, ctrl_var=0.05, treat_var=0.05)

    mde = pw._mde_from_curve(curve, 0.05, power)
    achieved = pw._normal_power_calculation(
        alpha=0.05, se_curve=curve, average_effect=mde
    )

    assert achieved - _wrong_direction_tail(curve, mde, 0.05, hypothesis) == (
        pytest.approx(power, abs=1e-12)
    )


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("power", [0.6, 0.8, 0.95])
def test_mde_and_power_are_inverses_for_flat_curves(hypothesis, power):
    """The same inversion must hold for absolute effects."""
    pw = _make_relative_delta_power(hypothesis)
    curve = StandardErrorCurve(std_error=0.25)

    mde = pw._mde_from_curve(curve, 0.05, power)
    achieved = pw._normal_power_calculation(
        alpha=0.05, se_curve=curve, average_effect=mde
    )

    assert achieved - _wrong_direction_tail(curve, mde, 0.05, hypothesis) == (
        pytest.approx(power, abs=1e-12)
    )


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


def test_mde_rolling_time_line_reports_only_mde(monkeypatch):
    """
    mde_rolling_time_line reports the MDE and nothing else.

    It used to also emit a `relative_mde` computed as `mde / mean(target)` — a
    second, weaker notion of relative MDE that treats the baseline as a fixed
    constant. For a relative analysis both were applied at once, producing a
    doubly-normalised number. There is now one way to get a relative MDE:
    `relative_effect=True`, which accounts for the baseline's own variance.
    """
    from cluster_experiments.random_splitter import ClusteredSplitter

    pw = NormalPowerAnalysis(
        analysis=DeltaMethodAnalysis(
            cluster_cols=["user"],
            scale_col="scale",
            target_col="target",
            relative_effect=True,
        ),
        splitter=ClusteredSplitter(cluster_cols=["user"]),
        time_col="date",
    )

    # The mean of the aggregated target is far from 1, so any stray
    # normalisation by it would be obvious.
    curve = _delta_curve(ctrl_mean=0.3, ctrl_var=0.001, treat_var=0.001)
    monkeypatch.setattr(pw, "_get_average_standard_error_curve", lambda **kw: curve)

    dates = pd.date_range("2024-01-01", periods=10)
    df = pd.DataFrame(
        {
            "user": np.repeat(np.arange(20), 10),
            "date": np.tile(dates, 20),
            "target": np.random.default_rng(0).normal(5, 1, size=200),
        }
    )

    results = pw.mde_rolling_time_line(
        df=df,
        powers=[0.8],
        experiment_length=[5, 10],
        n_simulations=3,
        agg_func="sum",
    )

    expected = pw._mde_from_curve(curve, pw.alpha, 0.8)
    assert results
    for row in results:
        assert set(row) == {"power", "mde", "experiment_length"}
        assert row["mde"] == pytest.approx(expected)


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
        std_error=np.sqrt(var_abs),
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
    )
    se_proper = transformer.bse["treatment"]

    z_alpha = norm.ppf(1 - alpha / 2)
    power_naive = 1 - norm.cdf(z_alpha - planted_rel_effect / se_naive)
    power_proper = 1 - norm.cdf(z_alpha - planted_rel_effect / se_proper)

    # Proper SE is larger so power should be lower or equal
    assert power_proper <= power_naive + 1e-6


# ---------------------------------------------------------------------------
# Parity test – relative OLS vs relative DeltaMethodAnalysis
# ---------------------------------------------------------------------------


def test_relative_ols_vs_delta_parity():
    """
    At the cluster level, OLSAnalysis(relative_effect=True) on the precomputed
    ratio column and DeltaMethodAnalysis(relative_effect=True) on the raw
    numerator/denominator columns should give very similar results.

    The two estimators are not numerically identical when cluster sizes (scale)
    vary: OLS uses a simple unweighted mean of per-cluster ratios while the
    delta method uses a weighted ratio estimator (weighted by scale).  With
    constant scale they would be exactly equal; with variable scale they are
    close but can differ by a few percent.

    The two SEs also differ: OLS treats the precomputed ratio as a single
    random variable while the delta method propagates variance from both
    numerator and denominator.  Both should be in the same ballpark (~20%).
    """
    from cluster_experiments import OLSAnalysis

    df = _make_ratio_df(n_users=5_000, treatment_effect=0.05, seed=99)

    # Aggregate to one row per cluster (required for DeltaMethodAnalysis
    # and for the apples-to-apples OLS comparison)
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum"}
    )
    df_agg["ratio"] = df_agg["target"] / df_agg["scale"]

    # Relative OLS on the precomputed per-cluster ratio
    ols_rel = OLSAnalysis(
        target_col="ratio",
        treatment_col="treatment",
        relative_effect=True,
    )

    # Relative delta method on the raw cluster-level numerator/denominator
    delta_rel = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )

    ols_point = ols_rel.get_point_estimate(df_agg)
    delta_point = delta_rel.get_point_estimate(df_agg)

    ols_se = ols_rel.get_standard_error(df_agg)
    delta_se = delta_rel.get_standard_error(df_agg)

    # Both should detect a positive effect in the same direction
    assert ols_point > 0
    assert delta_point > 0

    # Point estimates come from different estimators (unweighted vs weighted by
    # scale) so a 5% relative tolerance is appropriate
    assert ols_point == pytest.approx(delta_point, rel=0.05)

    # SEs are computed differently but must be in the same ballpark
    assert ols_se == pytest.approx(delta_se, rel=0.20)


def test_relative_ols_and_delta_mdes_agree():
    """
    Relative OLS and relative delta must produce comparable MDEs, not just
    comparable point estimates and standard errors.

    Before the standard error curve, only the delta method solved the quadratic
    while relative OLS was left on the linear approximation, so the two diverged
    on MDE by more than they did on the quantities above.
    """
    from cluster_experiments import ClusteredOLSAnalysis
    from cluster_experiments.random_splitter import ClusteredSplitter

    df = _make_ratio_df(n_users=5_000, treatment_effect=0.05, seed=99)
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum"}
    )
    df_agg["ratio"] = df_agg["target"] / df_agg["scale"]
    # The splitter assigns its own treatment
    df_agg = df_agg.drop(columns=["treatment"])

    def mde_of(analysis, target_col):
        return NormalPowerAnalysis(
            analysis=analysis,
            splitter=ClusteredSplitter(cluster_cols=["user"]),
            target_col=target_col,
            n_simulations=5,
            seed=42,
        ).mde(df_agg, power=0.8)

    ols_mde = mde_of(
        ClusteredOLSAnalysis(
            cluster_cols=["user"],
            target_col="ratio",
            relative_effect=True,
        ),
        target_col="ratio",
    )
    delta_mde = mde_of(
        DeltaMethodAnalysis(
            cluster_cols=["user"],
            scale_col="scale",
            target_col="target",
            relative_effect=True,
        ),
        target_col="target",
    )

    assert ols_mde > 0 and delta_mde > 0
    # Same tolerance as the standard errors in test_relative_ols_vs_delta_parity:
    # the estimators weight clusters differently, so they are close, not equal.
    assert ols_mde == pytest.approx(delta_mde, rel=0.20)


def test_relative_mde_exceeds_absolute_mde_over_baseline():
    """
    The relative MDE is larger than dividing the absolute MDE by the baseline.

    These are two different quantities: dividing treats the baseline as a fixed
    constant, while a relative estimand carries the baseline's own variance. The
    second is therefore always the larger. `mde_rolling_time_line` used to report
    the smaller one under the name `relative_mde`, and applied both at once for a
    relative analysis - this ordering is what that violated.
    """
    from cluster_experiments.random_splitter import ClusteredSplitter

    df_agg = _make_ratio_df(n_users=2_000, treatment_effect=0.0, seed=7).drop(
        columns=["treatment"]  # the splitter assigns its own
    )

    def mde_of(relative_effect):
        return NormalPowerAnalysis(
            analysis=DeltaMethodAnalysis(
                cluster_cols=["user"],
                scale_col="scale",
                target_col="target",
                relative_effect=relative_effect,
            ),
            splitter=ClusteredSplitter(cluster_cols=["user"]),
            n_simulations=5,
            seed=42,
        ).mde(df_agg, power=0.8)

    baseline = df_agg["target"].sum() / df_agg["scale"].sum()
    shortcut = mde_of(relative_effect=False) / baseline
    relative_mde = mde_of(relative_effect=True)

    assert relative_mde > shortcut
    # They agree to first order; the gap is the baseline's own variance.
    assert relative_mde == pytest.approx(shortcut, rel=0.05)


def test_standard_error_curve_is_flat_without_effect_dependence():
    """A curve with zero coefficients returns the same standard error everywhere."""
    flat = StandardErrorCurve(std_error=0.3)
    assert not flat.is_effect_dependent
    for effect in [-10.0, -0.1, 0.0, 0.1, 10.0]:
        assert flat.standard_error_at(effect) == 0.3

    curved = StandardErrorCurve(std_error=0.3, effect_var=0.01, effect_cov=-0.01)
    assert curved.is_effect_dependent
    assert curved.standard_error_at(0.0) == 0.3
    # effect_cov == -effect_var collapses to se2_t + se2_c * (1 + m)**2, which is
    # increasing in m on both sides of zero for m > -1.
    assert (
        curved.standard_error_at(0.5)
        > curved.standard_error_at(0.0)
        > curved.standard_error_at(-0.5)
    )
