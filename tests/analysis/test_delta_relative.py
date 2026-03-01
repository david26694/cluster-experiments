"""Tests for relative effects (point estimate, SE, power, MDE) with DeltaMethodAnalysis.
Modeled after tests/analysis/test_lift_transformer.py.
"""
from copy import deepcopy

import numpy as np
import pytest

from cluster_experiments import (
    AnalysisPlan,
    DeltaMethodAnalysis,
    NormalPowerAnalysis,
    PowerAnalysis,
)
from cluster_experiments.random_splitter import ClusteredSplitter
from tests.utils import generate_ratio_metric_data


def test_relative_delta_point_estimate(analysis_ratio_df, experiment_dates):
    """Relative point estimate should match naive (absolute_effect / ctrl_mean)."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    analyser_abs = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=False
    )
    analyser_rel = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )

    absolute_ate = analyser_abs.get_point_estimate(df)
    relative_ate = analyser_rel.get_point_estimate(df)

    # Control ratio mean for naive comparison
    df_binary = df.copy()
    df_binary["treatment"] = (df_binary["treatment"] == "B").astype(int)
    agg = df_binary.groupby(["user", "treatment"]).agg(
        {"target": "sum", "scale": "sum"}
    )
    ctrl = agg.loc[agg.index.get_level_values("treatment") == 0]
    ctrl_mean = ctrl["target"].sum() / ctrl["scale"].sum()

    assert relative_ate == pytest.approx(absolute_ate / ctrl_mean, rel=1e-4)


def test_relative_delta_se(analysis_ratio_df, experiment_dates):
    """Relative SE should be slightly bigger than naive (SE_abs / ctrl_mean)."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    analyser_abs = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=False
    )
    analyser_rel = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )

    se_abs = analyser_abs.get_standard_error(df)
    se_rel = analyser_rel.get_standard_error(df)

    df_binary = df.copy()
    df_binary["treatment"] = (df_binary["treatment"] == "B").astype(int)
    agg = df_binary.groupby(["user", "treatment"]).agg(
        {"target": "sum", "scale": "sum"}
    )
    ctrl = agg.loc[agg.index.get_level_values("treatment") == 0]
    ctrl_mean = ctrl["target"].sum() / ctrl["scale"].sum()
    naive_se_rel = se_abs / ctrl_mean

    assert se_rel >= naive_se_rel
    assert se_rel == pytest.approx(naive_se_rel, rel=5e-2)


def test_relative_delta_ci_and_pvalue(analysis_ratio_df, experiment_dates):
    """CI and p-value should be consistent: if p < alpha, CI should not contain 0."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )

    for alpha in [0.05, 0.01, 0.001]:
        ci = analyser.get_confidence_interval(df, alpha=alpha)
        p_value = analyser.get_pvalue(df)
        if p_value < alpha:
            assert ci.lower * ci.upper > 0
        else:
            assert ci.lower * ci.upper < 0


def test_relative_delta_power_lower_than_naive(analysis_ratio_df, experiment_dates):
    """Power with relative effect should be slightly lower than naive (larger SE)."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    splitter = ClusteredSplitter(cluster_cols=["user"])
    analysis_abs = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=False
    )
    analysis_rel = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )

    pw_abs = NormalPowerAnalysis(
        splitter=splitter, analysis=analysis_abs, n_simulations=30
    )
    pw_rel = NormalPowerAnalysis(
        splitter=splitter, analysis=analysis_rel, n_simulations=30
    )

    # Use a small effect so power is not 1
    np.random.seed(42)
    power_abs = pw_abs.power_analysis(df, average_effect=0.01)
    np.random.seed(42)
    power_rel = pw_rel.power_analysis(df, average_effect=0.01)

    # Relative SE is larger, so power should be <= absolute power
    assert power_rel <= power_abs + 0.05  # allow small Monte Carlo noise


def test_relative_ratio_mde_calculation_formula():
    """Quadratic MDE solver returns positive m for valid inputs (A, B, C_term formula)."""
    from cluster_experiments.random_splitter import ClusteredSplitter

    splitter = ClusteredSplitter(cluster_cols=["user"])
    analysis = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )
    pw = NormalPowerAnalysis(splitter=splitter, analysis=analysis)

    ctrl_mean, ctrl_var, treat_var = 1.0, 0.01, 0.01
    m = pw._relative_ratio_mde_calculation(
        alpha=0.05,
        ctrl_mean=ctrl_mean,
        ctrl_var=ctrl_var,
        treat_var=treat_var,
        power=0.8,
    )
    assert m > 0
    assert np.isfinite(m)


def test_relative_delta_mde_returns_positive(analysis_ratio_df, experiment_dates):
    """Relative MDE (dimensionless) should be positive and finite."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    splitter = ClusteredSplitter(cluster_cols=["user"])
    analysis = DeltaMethodAnalysis(
        cluster_cols=["user"], scale_col="scale", relative_effect=True
    )
    pw = NormalPowerAnalysis(
        splitter=splitter, analysis=analysis, n_simulations=20
    )

    mde_rel = pw.mde(df, power=0.8)
    assert mde_rel > 0
    assert np.isfinite(mde_rel)


def test_plan_config_delta_relative(analysis_ratio_df, experiment_dates):
    """AnalysisPlan with delta and relative_effect: relative ate matches naive."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    config_relative = {
        "metrics": [
            {
                "alias": "ratio",
                "numerator_name": "target",
                "denominator_name": "scale",
            },
        ],
        "variants": [
            {"name": "A", "is_control": True},
            {"name": "B", "is_control": False},
        ],
        "variant_col": "treatment",
        "analysis_type": "delta",
        "analysis_config": {"cluster_cols": ["user"], "relative_effect": True},
    }
    config_abs = deepcopy(config_relative)
    config_abs["analysis_config"] = {"cluster_cols": ["user"]}

    plan_rel = AnalysisPlan.from_metrics_dict(config_relative)
    plan_abs = AnalysisPlan.from_metrics_dict(config_abs)

    results_rel = plan_rel.analyze(df)
    results_abs = plan_abs.analyze(df)

    # Control ratio mean for naive comparison
    df_binary = df.copy()
    df_binary["treatment"] = (df_binary["treatment"] == "B").astype(int)
    agg = df_binary.groupby(["user", "treatment"]).agg(
        {"target": "sum", "scale": "sum"}
    )
    ctrl = agg.loc[agg.index.get_level_values("treatment") == 0]
    ctrl_mean = ctrl["target"].sum() / ctrl["scale"].sum()

    assert plan_rel.tests[0].experiment_analysis.relative_effect
    assert results_rel.ate[0] == pytest.approx(
        results_abs.ate[0] / ctrl_mean, rel=1e-4
    )


def test_relative_delta_raises_with_covariates(analysis_ratio_df, experiment_dates):
    """relative_effect with covariates should raise when running analysis (not supported yet)."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")
    df = df.copy()
    df["x1"] = np.random.randn(len(df))
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum", "x1": "mean"}
    )

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        relative_effect=True,
        covariates=["x1"],
    )
    with pytest.raises(ValueError, match="relative_effect.*without covariates"):
        analyser.get_pvalue(df_agg)


def test_get_ratio_mde_stats_raises_with_covariates(analysis_ratio_df, experiment_dates):
    """get_ratio_mde_stats with covariates should raise."""
    experiment_start = min(experiment_dates)
    df = analysis_ratio_df.query(f"date >= '{experiment_start}'")
    # Add a covariate
    df = df.copy()
    df["x1"] = np.random.randn(len(df))

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        covariates=["x1"],
        relative_effect=False,
    )
    # Aggregate so covariates are at cluster level
    df_agg = df.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum", "x1": "mean"}
    )

    with pytest.raises(ValueError, match="without covariates"):
        analyser.get_ratio_mde_stats(df_agg)
