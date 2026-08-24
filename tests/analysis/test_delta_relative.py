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


# ---------------------------------------------------------------------------
# Unit tests – DeltaMethodLiftTransformer directly
# ---------------------------------------------------------------------------


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


def test_relative_mde_raises_where_the_discriminant_is_still_positive():
    """
    The no-solution guard tests the leading coefficient, not the discriminant.

    Those are not the same condition. Cauchy-Schwarz bounds effect_cov**2 by
    std_error**2 * effect_var, so the discriminant stays non-negative for a while
    after the leading coefficient turns negative. In that window the quadratic
    still has two real roots, but neither solves m = k * SE(m): they solve
    m = -k * SE(m) instead. A discriminant test would let that through and return
    a large negative number for what is a perfectly ordinary alpha and power.

    The window needs a noisy baseline rather than an exotic design: at alpha=0.05
    and power=0.9 a relative standard error of 32% on the baseline lands in it.
    """
    ctrl_mean, ctrl_var, treat_var = 1.0, 0.10, 0.10
    alpha, power = 0.05, 0.9
    curve = _delta_curve(ctrl_mean, ctrl_var, treat_var)

    # confirm the fixture really is inside the window, not merely past the ceiling
    k = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    leading_coefficient = 1 - k**2 * curve.effect_var
    discriminant = (2 * k**2 * curve.effect_cov) ** 2 - 4 * leading_coefficient * (
        -(k**2) * curve.std_error**2
    )
    assert leading_coefficient < 0, "fixture should have no solution"
    assert discriminant > 0, "fixture should still have real roots"

    with pytest.raises(ValueError, match="No finite minimum detectable effect"):
        _make_relative_delta_power()._mde_from_curve(curve, alpha, power)


# ---------------------------------------------------------------------------
# Integration – DeltaMethodAnalysis(relative_effect=True)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parity test – relative OLS vs relative DeltaMethodAnalysis
# ---------------------------------------------------------------------------


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


def test_relative_delta_with_covariates_recovers_planted_lift():
    """
    A relative effect with CUPED covariates must recover the planted lift, and
    agree with the same analysis run without covariates.

    The covariate mean used to centre the CUPED correction has to be on the same
    scale as the covariate. Dividing the covariate sum by the scale sum instead
    leaves a constant offset in every corrected target. That offset cancels out
    of the treatment-control difference, so an absolute effect is unharmed, but it
    does not cancel out of a ratio: it lands in the denominator of the relative
    lift and inflated it by a factor of the mean scale (~19x here).
    """
    n_users, true_relative_lift = 20_000, 0.10
    rng = np.random.default_rng(3)
    scale = rng.integers(5, 20, size=n_users).astype(float)
    base_rate = np.clip(0.30 + rng.normal(0, 0.06, size=n_users), 0.05, 0.95)
    treatment_flag = rng.integers(0, 2, size=n_users)
    df = pd.DataFrame(
        {
            "user": np.arange(n_users),
            "scale": scale,
            "target": rng.binomial(
                scale.astype(int),
                np.clip(base_rate * (1 + true_relative_lift * treatment_flag), 0, 1),
            ).astype(float),
            "treatment": np.where(treatment_flag == 0, "A", "B"),
            # covariate on the ratio scale, as the delta-method CUPED expects
            "pre_rate": base_rate + rng.normal(0, 0.01, size=n_users),
        }
    )

    def relative_analysis(covariates):
        return DeltaMethodAnalysis(
            cluster_cols=["user"],
            scale_col="scale",
            target_col="target",
            covariates=covariates,
            relative_effect=True,
        )

    without = relative_analysis([])
    with_cuped = relative_analysis(["pre_rate"])

    lift_without = without.get_point_estimate(df)
    lift_with = with_cuped.get_point_estimate(df)

    # Both must land near the planted lift, and near each other.
    assert lift_without == pytest.approx(true_relative_lift, rel=0.20)
    assert lift_with == pytest.approx(true_relative_lift, rel=0.20)
    assert lift_with == pytest.approx(lift_without, rel=0.05)

    # CUPED must also tighten the relative standard error, not loosen it.
    assert with_cuped.get_standard_error(df) < without.get_standard_error(df)


def test_standard_error_curve_with_covariates():
    """
    The standard error curve is well formed for a relative delta analysis with
    covariates, and mde/power still invert each other exactly.
    """
    from cluster_experiments.random_splitter import ClusteredSplitter

    df = _make_ratio_df(
        n_users=4_000, treatment_effect=0.0, seed=21, with_covariate=True
    )
    covariates = ["pre_rate"]

    curve = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        covariates=covariates,
        relative_effect=True,
    ).get_standard_error_curve(df)

    assert curve.is_effect_dependent
    assert curve.std_error > 0
    # The arms stay independent under CUPED, so the covariance coefficient is
    # still exactly minus the variance one.
    assert curve.effect_cov == pytest.approx(-curve.effect_var, rel=1e-12)

    for hypothesis in ["greater", "less"]:
        pw = NormalPowerAnalysis(
            analysis=DeltaMethodAnalysis(
                cluster_cols=["user"],
                scale_col="scale",
                target_col="target",
                covariates=covariates,
                relative_effect=True,
                hypothesis=hypothesis,
            ),
            splitter=ClusteredSplitter(cluster_cols=["user"]),
        )
        mde = pw._mde_from_curve(curve, 0.05, 0.8)
        achieved = pw._normal_power_calculation(
            alpha=0.05, se_curve=curve, average_effect=mde
        )
        assert achieved == pytest.approx(0.8, abs=1e-12)


# ---------------------------------------------------------------------------
# Relative OLS through the standard error curve
# ---------------------------------------------------------------------------


def _relative_ols_setup(seed: int = 5, n: int = 2_000):
    """A clustered dataset plus a relative ClusteredOLSAnalysis with a covariate.

    Clustered OLS with a covariate is used deliberately. With plain OLS the
    regression algebra makes ``effect_cov`` equal ``-effect_var`` to within a
    rounding error, which is the delta-method special case; the cluster-robust
    covariance is what actually exercises the general three-coefficient form.
    """
    from cluster_experiments import ClusteredOLSAnalysis

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "target": rng.normal(10, 3, n),
            "x": rng.normal(0, 1, n),
            "cluster": rng.integers(0, 60, n),
            "treatment": np.where(rng.integers(0, 2, n) == 1, "B", "A"),
        }
    )
    df["target"] += 0.8 * df["x"]

    analysis = ClusteredOLSAnalysis(
        cluster_cols=["cluster"],
        target_col="target",
        covariates=["x"],
        relative_effect=True,
    )
    return df, analysis


def test_relative_ols_curve_is_not_the_delta_special_case():
    """
    The general three-coefficient form is genuinely exercised by relative OLS.

    For the delta method the arms are independent, forcing
    ``effect_cov == -effect_var``. Relative OLS has no such constraint: both the
    treatment coefficient and the adjusted control mean come out of one fit, so
    their covariance is whatever the regression says. If every test only ever saw
    ``effect_cov == -effect_var``, a mistake in the OLS coefficient extraction
    could not be distinguished from the delta case.
    """
    df, analysis = _relative_ols_setup()
    curve = analysis.get_standard_error_curve(df)

    assert curve.is_effect_dependent
    assert curve.effect_cov != pytest.approx(-curve.effect_var, rel=1e-3)


def test_relative_ols_curve_matches_regression_covariance():
    """
    The curve coefficients are the delta-method expansion of tau / mu_c, computed
    here independently from the regression covariance matrix.
    """
    import statsmodels.api as sm

    df, analysis = _relative_ols_setup()
    prepared = analysis._prepare(df)

    # An independent fit, so the reference does not reuse the production path.
    ols = sm.OLS.from_formula("target ~ treatment + x", data=prepared).fit(
        cov_type="cluster", cov_kwds={"groups": prepared["cluster"]}
    )

    covariance = ols.cov_params()
    treatment_effect = ols.params["treatment"]
    control_x_mean = prepared.loc[prepared["treatment"] == 0, "x"].mean()

    # mu_c, the covariate-adjusted control mean, and its variance
    adjusted_control_mean = ols.params["Intercept"] + control_x_mean * ols.params["x"]
    var_adjusted_control_mean = (
        covariance.loc["Intercept", "Intercept"]
        + control_x_mean**2 * covariance.loc["x", "x"]
        + 2 * control_x_mean * covariance.loc["Intercept", "x"]
    )
    cov_treatment_control_mean = (
        covariance.loc["treatment", "Intercept"]
        + control_x_mean * covariance.loc["treatment", "x"]
    )

    expected_std_error = np.sqrt(
        covariance.loc["treatment", "treatment"] / adjusted_control_mean**2
    )
    expected_effect_var = var_adjusted_control_mean / adjusted_control_mean**2
    expected_effect_cov = cov_treatment_control_mean / adjusted_control_mean**2

    curve = analysis.get_standard_error_curve(df)
    assert curve.std_error == pytest.approx(expected_std_error, rel=1e-12)
    assert curve.effect_var == pytest.approx(expected_effect_var, rel=1e-12)
    assert curve.effect_cov == pytest.approx(expected_effect_cov, rel=1e-12)

    # And the reported standard error is that curve evaluated at the observed lift
    observed_lift = treatment_effect / adjusted_control_mean
    assert analysis.get_standard_error(df) == pytest.approx(
        curve.standard_error_at(observed_lift), rel=1e-12
    )


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("power", [0.6, 0.8, 0.95])
def test_relative_ols_mde_and_power_are_inverses(hypothesis, power):
    """
    The MDE and power invert each other for relative OLS, not only for the delta
    method. Relative OLS is the estimator whose MDE changed value when it moved
    off the linear approximation, so it is the one that most needs this pinned.
    """
    from cluster_experiments import ClusteredOLSAnalysis
    from cluster_experiments.random_splitter import ClusteredSplitter

    df, _ = _relative_ols_setup()
    analysis = ClusteredOLSAnalysis(
        cluster_cols=["cluster"],
        target_col="target",
        covariates=["x"],
        relative_effect=True,
        hypothesis=hypothesis,
    )
    power_analysis = NormalPowerAnalysis(
        analysis=analysis, splitter=ClusteredSplitter(cluster_cols=["cluster"])
    )
    curve = analysis.get_standard_error_curve(df)

    mde = power_analysis._mde_from_curve(curve, 0.05, power)
    achieved = power_analysis._normal_power_calculation(
        alpha=0.05, se_curve=curve, average_effect=mde
    )

    assert achieved - _wrong_direction_tail(curve, mde, 0.05, hypothesis) == (
        pytest.approx(power, abs=1e-12)
    )


@pytest.mark.parametrize("hypothesis", ["two-sided", "greater"])
def test_relative_ols_mde_exceeds_the_linear_approximation(hypothesis):
    """
    The quadratic MDE is at least the linear one for relative OLS.

    This holds because ``effect_cov`` comes out negative here, so the standard
    error increases with the effect and a larger effect is needed. The sign is
    asserted rather than assumed: nothing in the regression algebra guarantees it,
    and if it flipped the inequality would legitimately reverse.
    """
    from cluster_experiments import ClusteredOLSAnalysis
    from cluster_experiments.random_splitter import ClusteredSplitter

    alpha, power = 0.05, 0.8
    df, _ = _relative_ols_setup()
    analysis = ClusteredOLSAnalysis(
        cluster_cols=["cluster"],
        target_col="target",
        covariates=["x"],
        relative_effect=True,
        hypothesis=hypothesis,
    )
    power_analysis = NormalPowerAnalysis(
        analysis=analysis, splitter=ClusteredSplitter(cluster_cols=["cluster"])
    )
    curve = analysis.get_standard_error_curve(df)
    assert curve.effect_cov < 0

    z_alpha = norm.ppf(1 - alpha / 2 if hypothesis == "two-sided" else 1 - alpha)
    linear_mde = (z_alpha + norm.ppf(power)) * curve.std_error

    assert power_analysis._mde_from_curve(curve, alpha, power) >= linear_mde


# ---------------------------------------------------------------------------
# DeltaMethodAnalysis vs ClusteredOLSAnalysis, on the same relative estimand
# ---------------------------------------------------------------------------


def test_relative_lift_handles_a_negative_baseline():
    """
    A ratio metric can be negative, so the curve must stay valid when the control
    mean is. The lift flips sign with the baseline, but the standard error is a
    magnitude and has to stay positive - which is why se_null divides by
    ``abs(ctrl_mean)``.
    """
    transformer = DeltaMethodLiftTransformer("treatment")
    transformer.fit(mean_diff=0.04, std_error=0.01, ctrl_mean=-0.50, ctrl_var=0.0001)
    curve = transformer.standard_error_curve()

    assert transformer.params["treatment"] < 0  # 0.04 / -0.50
    assert transformer.bse["treatment"] > 0
    assert curve.std_error > 0
    assert curve.effect_var > 0
    assert curve.effect_cov == pytest.approx(-curve.effect_var, rel=1e-12)
    # and the curve still reproduces the reported standard error
    assert curve.standard_error_at(transformer.params["treatment"]) == pytest.approx(
        transformer.bse["treatment"], rel=1e-12
    )


def _unaggregated_ratio_df(
    seed: int = 31, n_users: int = 800, with_covariate: bool = False
) -> pd.DataFrame:
    """
    One row per observation, not per cluster, with deliberately uneven cluster
    sizes.

    This is the representation in which the delta method and clustered OLS
    estimate the same thing. OLS over raw rows gives the difference in row-level
    means, which for a 0/1 outcome and one row per trial is exactly the ratio
    metric; the delta method reaches the same estimand by aggregating each cluster
    to (sum of target, count). Neither is reweighted by hand, so any disagreement
    is a real disagreement.
    """
    rng = np.random.default_rng(seed)
    cluster_sizes = rng.integers(1, 25, n_users)
    treatment_flag = rng.integers(0, 2, n_users)
    base_rate = np.clip(0.30 + rng.normal(0, 0.06, n_users), 0.05, 0.95)

    frames = []
    for user in range(n_users):
        rate = min(base_rate[user] * (1 + 0.08 * treatment_flag[user]), 1.0)
        frame = pd.DataFrame(
            {
                "user": user,
                "treatment": "B" if treatment_flag[user] else "A",
                "target": rng.binomial(1, rate, cluster_sizes[user]).astype(float),
                "scale": 1.0,
            }
        )
        if with_covariate:
            frame["pre"] = base_rate[user] + rng.normal(0, 0.02, cluster_sizes[user])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_delta_and_clustered_ols_agree_on_unaggregated_data():
    """
    The delta method and clustered OLS agree on the relative lift, its standard
    error, and the standard error curve, when both are given the same
    un-aggregated data.

    Both estimate the difference in row-level means expressed relative to control.
    Cluster sizes here range from 1 to 24, and no reweighting is applied to either
    side: OLS gets the weighting implicitly from having one row per observation,
    the delta method from aggregating to (sum, count). They are two independent
    implementations of the same estimand, which makes this the sharpest available
    cross-check on the relative-lift machinery.
    """
    from cluster_experiments import ClusteredOLSAnalysis

    df = _unaggregated_ratio_df()
    delta = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        target_col="target",
        relative_effect=True,
    )
    ols = ClusteredOLSAnalysis(
        cluster_cols=["user"], target_col="target", relative_effect=True
    )

    # the point estimate is the same number, not merely a close one
    assert delta.get_point_estimate(df) == pytest.approx(
        ols.get_point_estimate(df), rel=1e-9
    )
    # the standard errors differ only in how clustering is handled: analytically
    # for the delta method, cluster-robust sandwich for OLS
    assert delta.get_standard_error(df) == pytest.approx(
        ols.get_standard_error(df), rel=5e-3
    )

    delta_curve = delta.get_standard_error_curve(df)
    ols_curve = ols.get_standard_error_curve(df)
    assert delta_curve.std_error == pytest.approx(ols_curve.std_error, rel=5e-3)
    assert delta_curve.effect_var == pytest.approx(ols_curve.effect_var, rel=1e-2)
    assert delta_curve.effect_cov == pytest.approx(ols_curve.effect_cov, rel=1e-2)


def test_delta_and_clustered_ols_relative_mdes_agree_on_unaggregated_data():
    """
    The two estimators also agree on the relative MDE, which is the quantity this
    branch changed. Before the standard error curve, only the delta method solved
    the effect-dependent equation while relative OLS used the linear
    approximation, so their MDEs diverged by more than their standard errors did.
    """
    from cluster_experiments import ClusteredOLSAnalysis
    from cluster_experiments.random_splitter import ClusteredSplitter

    df = _unaggregated_ratio_df()

    def relative_mde(analysis):
        return NormalPowerAnalysis(
            analysis=analysis,
            splitter=ClusteredSplitter(cluster_cols=["user"]),
            n_simulations=5,
            seed=3,
        ).mde(df.drop(columns=["treatment"]), power=0.8)

    delta_mde = relative_mde(
        DeltaMethodAnalysis(
            cluster_cols=["user"],
            scale_col="scale",
            target_col="target",
            relative_effect=True,
        )
    )
    ols_mde = relative_mde(
        ClusteredOLSAnalysis(
            cluster_cols=["user"], target_col="target", relative_effect=True
        )
    )

    assert delta_mde > 0
    assert delta_mde == pytest.approx(ols_mde, rel=1e-2)
