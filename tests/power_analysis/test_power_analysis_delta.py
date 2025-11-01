import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from cluster_experiments.experiment_analysis import DeltaMethodAnalysis
from cluster_experiments.power_analysis import NormalPowerAnalysis, PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter


@pytest.fixture
def delta_df_aggregated(delta_df):
    # Aggregates the delta_df at the cluster level
    return (
        delta_df.groupby(["user", "date"])
        .agg(
            {
                "treatment": "first",
                "x1": "mean",
                "x2": "mean",
                "target": "sum",
                "scale": "sum",
            }
        )
        .reset_index()
    )


def test_delta(delta_df):
    # simple power analysis for delta method
    config = dict(
        cluster_cols=["user", "date"],
        scale_col="scale",
        analysis="delta",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)

    delta_df = delta_df.drop(columns=["treatment"])

    power = pw.power_analysis(delta_df, average_effect=0.0)
    assert power >= 0
    assert power <= 1


def test_delta_covariate_raises(delta_df):
    # with covariates, should raise with non-aggregated data
    with pytest.raises(
        ValueError,
        match="The data should be aggregated at the cluster level for the Delta Method analysis using covariates",
    ):
        config = dict(
            cluster_cols=["user", "date"],
            scale_col="scale",
            analysis="delta",
            perturbator="constant",
            splitter="clustered",
            covariates=["x1", "x2"],
            n_simulations=4,
        )
        pw = PowerAnalysis.from_dict(config)

        delta_df = delta_df.drop(columns=["treatment"])

        power = pw.power_analysis(delta_df, average_effect=0.0)
        assert power >= 0
        assert power <= 1


def test_delta_covariate(delta_df_aggregated):
    config = dict(
        cluster_cols=["user", "date"],
        scale_col="scale",
        analysis="delta",
        perturbator="constant",
        splitter="clustered",
        covariates=["x1", "x2"],
        n_simulations=4,
    )
    delta_df_aggregated = delta_df_aggregated.drop(columns=["treatment"])
    for cls in [PowerAnalysis, NormalPowerAnalysis]:
        pw = cls.from_dict(config)
        power = pw.power_analysis(delta_df_aggregated, average_effect=0.0)
        assert power >= 0
        assert power <= 1


def test_delta_cuped(delta_df_aggregated):
    config = dict(
        cluster_cols=["user", "date"],
        scale_col="scale",
        analysis="delta",
        perturbator="constant",
        splitter="clustered",
        covariates=["estimate_target"],
        agg_col="user",
        cupac_model="mean_cupac_model",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)

    delta_df = delta_df_aggregated.drop(columns=["treatment"])
    pre_delta_df = delta_df_aggregated.copy()

    for cls in [PowerAnalysis, NormalPowerAnalysis]:
        pw = cls.from_dict(config)
        power = pw.power_analysis(delta_df, pre_delta_df, average_effect=0.0)
        assert power >= 0
        assert power <= 1


def test_delta_cupac(delta_df_aggregated):
    splitter = ClusteredSplitter(
        cluster_cols=["user", "date"],
    )
    analysis = DeltaMethodAnalysis(
        cluster_cols=["user", "date"],
        covariates=["estimate_target"],
    )

    power = NormalPowerAnalysis(
        splitter=splitter,
        analysis=analysis,
        cupac_model=HistGradientBoostingRegressor(max_iter=3),
        features_cupac_model=["x1", "x2"],
        target_col="target",
        scale_col="scale",
        n_simulations=3,
    )
    delta_df = delta_df_aggregated.drop(columns=["treatment"])
    pre_delta_df = delta_df_aggregated.copy()

    power_result = power.power_analysis(delta_df, pre_delta_df, average_effect=0.0)
    assert power_result >= 0
