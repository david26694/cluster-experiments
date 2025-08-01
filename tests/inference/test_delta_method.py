import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from cluster_experiments import AnalysisPlan, HypothesisTest, RatioMetric, Variant


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


def test_from_metrics_dict(delta_df):
    # Create analysis plan
    plan = AnalysisPlan.from_metrics_dict(
        {
            "metrics": [
                {
                    "alias": "target",
                    "numerator_name": "target",
                    "denominator_name": "scale",
                },
            ],
            "variants": [
                {"name": "A", "is_control": True},
                {"name": "B", "is_control": False},
            ],
            "variant_col": "treatment",
            "alpha": 0.05,
            "analysis_type": "delta",
            "analysis_config": {"cluster_cols": ["user"]},
        }
    )
    # Run analysis
    results = plan.analyze(delta_df).to_dataframe()

    # Check results
    assert results["analysis_type"].iloc[0] == "delta"


def test_from_metrics_dict_covariate(delta_df_aggregated):
    # Create analysis plan
    plan = AnalysisPlan.from_metrics_dict(
        {
            "metrics": [
                {
                    "alias": "target",
                    "numerator_name": "target",
                    "denominator_name": "scale",
                },
            ],
            "variants": [
                {"name": "A", "is_control": True},
                {"name": "B", "is_control": False},
            ],
            "variant_col": "treatment",
            "alpha": 0.05,
            "analysis_type": "delta",
            "analysis_config": {
                "cluster_cols": ["user", "date"],
                "covariates": ["x1", "x2"],
            },
        }
    )
    # Run analysis
    results = plan.analyze(delta_df_aggregated).to_dataframe()

    # Check results
    assert results["analysis_type"].iloc[0] == "delta"


def test_from_metrics_dict_cuped(delta_df_aggregated):
    # Create analysis plan
    hypothesis_test = HypothesisTest(
        metric=RatioMetric(
            alias="target", numerator_name="target", denominator_name="scale"
        ),
        analysis_type="delta",
        analysis_config={
            "cluster_cols": ["user", "date"],
            "covariates": ["estimate_target"],
        },
        cupac_config={
            "cupac_model": HistGradientBoostingRegressor(max_iter=3),
            "features_cupac_model": ["x1", "x2"],
        },
    )

    plan = AnalysisPlan(
        tests=[hypothesis_test],
        variants=[
            Variant("A", is_control=True),
            Variant("B", is_control=False),
        ],
        variant_col="treatment",
    )

    # Run analysis
    pre_experiment_df = delta_df_aggregated[delta_df_aggregated["date"] < "2022-01-30"]
    experiment_df = delta_df_aggregated[delta_df_aggregated["date"] >= "2022-01-30"]
    results = plan.analyze(experiment_df, pre_experiment_df).to_dataframe()
    # Check results
    assert results["analysis_type"].iloc[0] == "delta"
