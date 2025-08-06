import numpy as np
import pytest

from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    DeltaMethodAnalysis,
)
from tests.utils import (
    generate_ratio_metric_data,
)


def test_delta_analysis(analysis_ratio_df, experiment_dates):
    analyser = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")
    experiment_start = min(experiment_dates)

    df_experiment = analysis_ratio_df.query(f"date >= '{experiment_start}'")

    assert 0.05 >= analyser.get_pvalue(df_experiment) >= 0


def test_aa_delta_analysis(dates):
    analyser = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")
    np.random.seed(2024)
    p_values = []
    for _ in range(1000):
        data = generate_ratio_metric_data(
            dates, 40_000, num_users=5000, treatment_effect=0
        )
        p_values.append(analyser.get_pvalue(data))

    positive_rate = sum(p < 0.05 for p in p_values) / len(p_values)

    assert positive_rate == pytest.approx(
        0.05, abs=0.01
    ), "P-value A/A calculation is incorrect"


def test_delta_analysis_aggregation(dates):
    analyser = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")

    # Generate data without effect so pvalues are not zero
    np.random.seed(2024)
    df_experiment = generate_ratio_metric_data(
        dates, 40_000, num_users=5000, treatment_effect=0
    )

    df_experiment_aggregated = df_experiment.groupby(
        ["user", "treatment"], as_index=False
    ).agg({"target": "sum", "scale": "sum"})

    pvalue = analyser.get_pvalue(df_experiment)
    pvalue_agg = analyser.get_pvalue(df_experiment_aggregated)
    point_estimate = analyser.get_point_estimate(df_experiment)
    point_estimate_agg = analyser.get_point_estimate(df_experiment_aggregated)

    assert pvalue == pytest.approx(
        pvalue_agg, rel=1e-8
    ), "Aggregation method is not working properly"
    assert point_estimate == pytest.approx(
        point_estimate_agg, rel=1e-8
    ), "Aggregation method is not working properly"


def test_stats_delta_vs_ols(analysis_ratio_df, experiment_dates):
    np.random.seed(2024)
    experiment_start_date = min(experiment_dates)

    analyser_ols = ClusteredOLSAnalysis(cluster_cols=["user"])
    analyser_delta = DeltaMethodAnalysis(cluster_cols=["user"], scale_col="scale")
    df = analysis_ratio_df.query(f"date >= '{experiment_start_date}'")

    point_estimate_ols = analyser_ols.get_point_estimate(df)
    point_estimate_delta = analyser_delta.get_point_estimate(df)

    SE_ols = analyser_ols.get_standard_error(df)
    SE_delta = analyser_delta.get_standard_error(df)

    assert point_estimate_delta == pytest.approx(
        point_estimate_ols, rel=1e-3
    ), "Point estimate is not consistent with Clustered OLS"
    assert SE_delta == pytest.approx(
        SE_ols, rel=1e-3
    ), "Standard error is not consistent with Clustered OLS"


def test_delta_cuped_analysis(analysis_ratio_df, experiment_dates):
    experiment_start = min(experiment_dates)

    df_experiment = analysis_ratio_df.query(f"date >= '{experiment_start}'")
    df_pre_experiment = analysis_ratio_df.query(f"date < '{experiment_start}'")
    df_experiment = df_experiment.groupby(["user", "treatment"], as_index=False).agg(
        {"target": "sum", "scale": "sum", "user_target_means": "mean"}
    )
    df_pre_experiment = df_pre_experiment.groupby(
        ["user", "treatment"], as_index=False
    ).agg(
        pre_target=("target", "sum"),
        pre_scale=("scale", "sum"),
    )
    df = df_experiment.merge(df_pre_experiment, on=["user", "treatment"])

    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        covariates=["user_target_means"],
    )

    assert 0.05 >= analyser.get_pvalue(df) >= 0


def test_aa_delta_cuped_analysis(dates):
    analyser = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        covariates=["pre_user_target_mean"],
    )
    np.random.seed(2024)
    p_values = []
    for _ in range(1000):
        data = generate_ratio_metric_data(
            dates, 40_000, num_users=5000, treatment_effect=0
        )
        pre_data = generate_ratio_metric_data(
            dates, 40_000, num_users=5000, treatment_effect=0
        )

        data = data.groupby(["user", "treatment"], as_index=False).agg(
            {"target": "sum", "scale": "sum"}
        )
        pre_data = pre_data.groupby(["user", "treatment"], as_index=False).agg(
            pre_target=("target", "sum"),
            pre_scale=("scale", "sum"),
            pre_user_target_mean=("user_target_means", "mean"),
        )
        data = data.merge(pre_data, on=["user", "treatment"])
        p_values.append(analyser.get_pvalue(data))

    positive_rate = sum(p < 0.05 for p in p_values) / len(p_values)

    assert positive_rate == pytest.approx(
        0.05, abs=0.015
    ), "P-value A/A calculation is incorrect"


def test_stats_delta_cuped_vs_ols(analysis_ratio_df_large, experiment_dates):
    np.random.seed(2024)

    experiment_start_date = min(experiment_dates)

    analyser_ols = ClusteredOLSAnalysis(
        cluster_cols=["user"], covariates=["pre_user_target_mean"]
    )
    analyser_delta = DeltaMethodAnalysis(
        cluster_cols=["user"],
        scale_col="scale",
        covariates=["pre_user_target_mean"],
    )

    df_experiment = analysis_ratio_df_large.query(f"date >= '{experiment_start_date}'")
    df_pre_experiment = analysis_ratio_df_large.query(
        f"date < '{experiment_start_date}'"
    )

    df_pre_experiment = df_pre_experiment.groupby(
        ["user", "treatment"], as_index=False
    ).agg(
        pre_target=("target", "sum"),
        pre_scale=("scale", "sum"),
        pre_user_target_mean=("user_target_means", "mean"),
    )
    df = df_experiment.merge(df_pre_experiment, on=["user", "treatment"])
    # impute nans with mean values
    df["pre_target"] = df["pre_target"].fillna(df["pre_target"].mean())
    df["pre_scale"] = df["pre_scale"].fillna(df["pre_scale"].mean())
    df["pre_user_target_mean"] = df["pre_user_target_mean"].fillna(
        df["pre_user_target_mean"].mean()
    )

    df_delta = df.groupby(["user", "treatment"], as_index=False).agg(
        {
            "target": "sum",
            "scale": "sum",
            "pre_target": "mean",
            "pre_scale": "mean",
            "pre_user_target_mean": "mean",
        }
    )

    point_estimate_ols = analyser_ols.get_point_estimate(df)
    point_estimate_delta = analyser_delta.get_point_estimate(df_delta)

    SE_ols = analyser_ols.get_standard_error(df)
    SE_delta = analyser_delta.get_standard_error(df_delta)

    assert point_estimate_delta == pytest.approx(
        point_estimate_ols, rel=5e-2
    ), "Point estimate is not consistent with Clustered OLS"
    assert SE_delta == pytest.approx(
        SE_ols, rel=5e-2
    ), "Standard error is not consistent with Clustered OLS"
