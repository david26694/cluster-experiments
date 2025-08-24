import numpy as np
import pandas as pd
import pytest

from cluster_experiments import AnalysisPlan, ClusteredSplitter, ConstantPerturbator


@pytest.fixture
def data_generation_params():
    """Parameters for data generation matching the notebook"""
    return {
        "N": 20000,  # Smaller for tests
        "NUM_CUSTOMERS": 5000,
        "seed": 42,
    }


@pytest.fixture
def customers(data_generation_params):
    """Generate customers data matching notebook logic"""
    np.random.seed(data_generation_params["seed"])

    num_customers = data_generation_params["NUM_CUSTOMERS"]
    customer_ids = np.arange(1, num_customers + 1)
    customer_ages = np.random.randint(20, 60, size=num_customers)
    customer_historical_orders = np.random.poisson(5, size=num_customers)
    mean_order_values = (
        50
        + 2.5 * customer_ages
        - 2 * (customer_ages <= 30)
        + 2 * (customer_historical_orders >= 8)
        + np.random.normal(0, 15, size=num_customers)
    )

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "customer_age": customer_ages,
            "mean_order_value": mean_order_values,
            "historical_orders": customer_historical_orders,
        }
    )


@pytest.fixture
def experiment_data(customers, data_generation_params):
    """Generate experiment data matching notebook"""
    np.random.seed(data_generation_params["seed"])

    num_orders = data_generation_params["N"]
    date_range = pd.date_range(start="2024-01-01", end="2024-03-31")

    # Sample customers
    sampled_customers = np.random.choice(customers["customer_id"], size=num_orders)
    orders = pd.DataFrame({"customer_id": sampled_customers}).merge(
        customers, on="customer_id", how="left"
    )

    # Generate orders
    orders = orders.assign(
        order_value=lambda df: df["mean_order_value"]
        + np.random.normal(0, 15, size=len(df)),
        delivery_time=lambda df: 8
        + np.sin(df["customer_id"] / 10)
        + np.random.normal(0, 0.5, size=len(df)),
        city=lambda df: np.random.choice(["NYC", "LA"], size=len(df)),
        date=lambda df: np.random.choice(date_range, size=len(df)),
        one=1,
        day_of_week=lambda df: df["date"].dt.dayofweek,
        age_smaller_30=lambda df: (df["customer_age"] < 30).astype(int),
        historical_orders_smaller_8=lambda df: (df["historical_orders"] < 8).astype(
            int
        ),
    ).drop(columns=["mean_order_value"])

    return orders


@pytest.fixture
def perturbated_data(experiment_data):
    """Split into experiment and pre-experiment data with treatment effects"""
    experiment = experiment_data.query("date >= '2024-03-01' and date < '2024-03-21'")

    # Add treatment assignment and effects
    splitter = ClusteredSplitter(cluster_cols=["customer_id"])
    experiment = splitter.assign_treatment_df(experiment)

    # Add effects
    perturbator_ov = ConstantPerturbator(target_col="order_value")
    experiment = perturbator_ov.perturbate(experiment, average_effect=1.5)

    perturbator_dt = ConstantPerturbator(target_col="delivery_time")
    experiment = perturbator_dt.perturbate(experiment, average_effect=0.2)

    return experiment


@pytest.fixture
def raw_and_aggregated_data(perturbated_data):
    """Aggregate data to cluster level"""
    raw_data = perturbated_data

    agg_data = (
        raw_data.groupby("customer_id")
        .agg(
            {
                "order_value": "sum",
                "one": "sum",
                "delivery_time": "sum",
                "customer_age": "mean",
                "age_smaller_30": "mean",
                "historical_orders": "mean",
                "historical_orders_smaller_8": "mean",
                "day_of_week": "mean",
                "date": "min",
                "treatment": "first",
                "city": "first",
            }
        )
        .reset_index()
    )

    return raw_data, agg_data


def test_delta_vs_ols_no_covariates(raw_and_aggregated_data):
    """Test that Delta Method matches Clustered OLS without covariates"""
    raw_data, agg_data = raw_and_aggregated_data

    # Delta Method Analysis
    results_delta = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {
                        "alias": "AOV",
                        "numerator_name": "order_value",
                        "denominator_name": "one",
                    },
                    {
                        "alias": "delivery_time",
                        "numerator_name": "delivery_time",
                        "denominator_name": "one",
                    },
                ],
                "variants": [
                    {"name": "A", "is_control": True},
                    {"name": "B", "is_control": False},
                ],
                "variant_col": "treatment",
                "alpha": 0.05,
                "analysis_type": "delta",
                "analysis_config": {"cluster_cols": ["customer_id"]},
            }
        )
        .analyze(raw_data)
        .to_dataframe()
    )

    # Clustered OLS Analysis
    results_ols = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {"alias": "AOV", "name": "order_value"},
                    {"alias": "delivery_time", "name": "delivery_time"},
                ],
                "variants": [
                    {"name": "A", "is_control": True},
                    {"name": "B", "is_control": False},
                ],
                "variant_col": "treatment",
                "alpha": 0.05,
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
            }
        )
        .analyze(raw_data)
        .to_dataframe()
    )

    # Check ATE matches (within 1%)
    ate_diff = (results_delta["ate"] - results_ols["ate"]).abs() / results_delta["ate"]
    assert (ate_diff < 0.01).all(), f"ATE difference too large: {ate_diff.max():.3f}"

    # Check standard error matches (within 1%)
    se_diff = (
        results_delta["std_error"] - results_ols["std_error"]
    ).abs() / results_delta["std_error"]
    assert (
        se_diff < 0.01
    ).all(), f"Standard error difference too large: {se_diff.max():.3f}"


def test_delta_vs_ols_single_covariate(raw_and_aggregated_data):
    """Test Delta Method vs OLS with single covariate"""
    raw_data, agg_data = raw_and_aggregated_data

    covariates = ["customer_age"]

    # Delta Method with covariates
    results_delta = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {
                        "alias": "AOV",
                        "numerator_name": "order_value",
                        "denominator_name": "one",
                    },
                    {
                        "alias": "delivery_time",
                        "numerator_name": "delivery_time",
                        "denominator_name": "one",
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
                    "cluster_cols": ["customer_id"],
                    "covariates": covariates,
                },
            }
        )
        .analyze(agg_data)
        .to_dataframe()
    )

    # Clustered OLS with covariates
    results_ols = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {"alias": "AOV", "name": "order_value"},
                    {"alias": "delivery_time", "name": "delivery_time"},
                ],
                "variants": [
                    {"name": "A", "is_control": True},
                    {"name": "B", "is_control": False},
                ],
                "variant_col": "treatment",
                "alpha": 0.05,
                "analysis_type": "clustered_ols",
                "analysis_config": {
                    "cluster_cols": ["customer_id"],
                    "covariates": covariates,
                },
            }
        )
        .analyze(raw_data)
        .to_dataframe()
    )

    # Check results are close (allowing more tolerance for covariate adjustment)
    ate_diff = (results_delta["ate"] - results_ols["ate"]).abs() / results_delta["ate"]
    assert (ate_diff < 0.01).all(), f"ATE difference too large: {ate_diff.max():.3f}"

    se_diff = (
        results_delta["std_error"] - results_ols["std_error"]
    ).abs() / results_delta["std_error"]
    assert (
        se_diff < 0.01
    ).all(), f"Standard error difference too large: {se_diff.max():.3f}"


def test_delta_vs_ols_multiple_covariates(raw_and_aggregated_data):
    """Test Delta Method vs OLS with multiple covariates"""
    raw_data, agg_data = raw_and_aggregated_data

    covariates = [
        "customer_age",
        "day_of_week",
        "age_smaller_30",
        "historical_orders_smaller_8",
    ]

    # Delta Method with multiple covariates
    results_delta = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {
                        "alias": "AOV",
                        "numerator_name": "order_value",
                        "denominator_name": "one",
                    },
                    {
                        "alias": "delivery_time",
                        "numerator_name": "delivery_time",
                        "denominator_name": "one",
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
                    "cluster_cols": ["customer_id"],
                    "covariates": covariates,
                },
            }
        )
        .analyze(agg_data)
        .to_dataframe()
    )

    # Clustered OLS with multiple covariates
    results_ols = (
        AnalysisPlan.from_metrics_dict(
            {
                "metrics": [
                    {"alias": "AOV", "name": "order_value"},
                    {"alias": "delivery_time", "name": "delivery_time"},
                ],
                "variants": [
                    {"name": "A", "is_control": True},
                    {"name": "B", "is_control": False},
                ],
                "variant_col": "treatment",
                "alpha": 0.05,
                "analysis_type": "clustered_ols",
                "analysis_config": {
                    "cluster_cols": ["customer_id"],
                    "covariates": covariates,
                },
            }
        )
        .analyze(raw_data)
        .to_dataframe()
    )

    # Check results are reasonably close
    ate_diff = (results_delta["ate"] - results_ols["ate"]).abs() / results_delta["ate"]
    assert (ate_diff < 0.01).all(), f"ATE difference too large: {ate_diff.max():.3f}"

    se_diff = (
        results_delta["std_error"] - results_ols["std_error"]
    ).abs() / results_delta["std_error"]
    assert (
        se_diff < 0.01
    ).all(), f"Standard error difference too large: {se_diff.max():.3f}"
