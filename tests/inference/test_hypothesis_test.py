import numpy as np
import pandas as pd
import pytest

from cluster_experiments import TargetAggregation
from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis
from cluster_experiments.inference.dimension import Dimension
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import SimpleMetric
from cluster_experiments.inference.variant import Variant
from cluster_experiments.power_config import analysis_mapping

# Set up constants for the data
NUM_ORDERS = 10000
NUM_CUSTOMERS = 3000
EXPERIMENT_GROUPS = ["control", "treatment_1", "treatment_2"]
GROUP_SIZE = NUM_CUSTOMERS // len(EXPERIMENT_GROUPS)


@pytest.fixture
def experiment_data():
    """Creates a realistic DataFrame for experimentation, along with pre-experiment data."""
    # Set up reproducible random data
    np.random.seed(42)
    customer_ids = np.arange(1, NUM_CUSTOMERS + 1)
    np.random.shuffle(customer_ids)

    # Create group mapping and orders
    experiment_group = np.repeat(EXPERIMENT_GROUPS, GROUP_SIZE)
    experiment_group = np.concatenate(
        (
            experiment_group,
            np.random.choice(EXPERIMENT_GROUPS, NUM_CUSTOMERS - len(experiment_group)),
        )
    )
    customer_group_mapping = dict(zip(customer_ids, experiment_group))

    order_ids = np.arange(1, NUM_ORDERS + 1)
    customers = np.random.choice(customer_ids, NUM_ORDERS)
    order_values = np.abs(np.random.normal(loc=10, scale=2, size=NUM_ORDERS))
    order_delivery_times = np.abs(np.random.normal(loc=30, scale=5, size=NUM_ORDERS))
    order_city_codes = np.random.randint(1, 3, NUM_ORDERS).astype(str)

    # Build experimental and pre-experiment data
    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": customers,
            "experiment_group": [
                customer_group_mapping[customer_id] for customer_id in customers
            ],
            "order_value": order_values,
            "order_delivery_time_in_minutes": order_delivery_times,
            "order_city_code": order_city_codes,
        }
    )
    pre_exp_df = df.assign(
        order_value=lambda df: df["order_value"]
        + np.random.normal(loc=0, scale=1, size=NUM_ORDERS),
        order_delivery_time_in_minutes=lambda df: df["order_delivery_time_in_minutes"]
        + np.random.normal(loc=0, scale=2, size=NUM_ORDERS),
    ).sample(int(NUM_ORDERS / 3))

    return df, pre_exp_df


@pytest.fixture
def variants():
    """Set up test variants for control and treatment groups."""
    return [
        Variant("control", is_control=True),
        Variant("treatment_1", is_control=False),
        Variant("treatment_2", is_control=False),
    ]


@pytest.fixture
def dimensions():
    """Sets up Dimension instance for testing."""
    return [Dimension(name="order_city_code", values=["1", "2"])]


@pytest.fixture
def metrics():
    """Sets up Metric instances for testing."""
    return {
        "order_value": SimpleMetric(alias="AOV", name="order_value"),
        "delivery_time": SimpleMetric(
            alias="AVG DT", name="order_delivery_time_in_minutes"
        ),
    }


@pytest.fixture
def cupac_config():
    """Sets up CUPAC configuration for testing with TargetAggregation model."""
    return {
        "cupac_model": TargetAggregation(
            agg_col="customer_id", target_col="order_delivery_time_in_minutes"
        ),
        "target_col": "order_delivery_time_in_minutes",
    }


def test_hypothesis_test_order_value(experiment_data, metrics, dimensions, variants):
    """Tests hypothesis testing on order value without CUPAC configuration."""
    df, pre_exp_df = experiment_data
    test_order_value = HypothesisTest(
        metric=metrics["order_value"],
        analysis_type="clustered_ols",
        analysis_config={"cluster_cols": ["customer_id"]},
        dimensions=dimensions,
    )

    # Test across both dimensions
    for dim in dimensions:
        for value in dim.values:
            test_results = test_order_value.get_test_results(
                control_variant=variants[0],
                treatment_variant=variants[1],
                variant_col="experiment_group",
                exp_data=df,
                dimension=dim,
                dimension_value=value,
                alpha=0.05,
            )

            assert test_results.metric_alias[0] == metrics["order_value"].alias
            assert test_results.control_variant_name[0] == variants[0].name
            assert test_results.treatment_variant_name[0] == variants[1].name
            assert isinstance(test_results.ate[0], float)
            assert 0 <= test_results.p_value[0] <= 1


def test_hypothesis_test_with_cupac(
    experiment_data, metrics, dimensions, variants, cupac_config
):
    """Tests hypothesis testing on delivery time using CUPAC configuration."""
    df, pre_exp_df = experiment_data
    test_delivery_time = HypothesisTest(
        metric=metrics["delivery_time"],
        analysis_type="gee",
        analysis_config={
            "cluster_cols": ["customer_id"],
            "covariates": ["estimate_order_delivery_time_in_minutes"],
        },
        cupac_config=cupac_config,
        dimensions=dimensions,
    )

    # Add covariates using CUPAC configuration
    exp_data_with_covariates = test_delivery_time.add_covariates(
        exp_data=df, pre_exp_data=pre_exp_df
    )

    # Test across both dimensions
    for dim in dimensions:
        for value in dim.values:
            test_results = test_delivery_time.get_test_results(
                control_variant=variants[0],
                treatment_variant=variants[1],
                variant_col="experiment_group",
                exp_data=exp_data_with_covariates,
                dimension=dim,
                dimension_value=value,
                alpha=0.05,
            )

            assert test_results.metric_alias[0] == metrics["delivery_time"].alias
            assert test_results.control_variant_name[0] == variants[0].name
            assert test_results.treatment_variant_name[0] == variants[1].name
            assert isinstance(test_results.ate[0], float)
            assert 0 <= test_results.p_value[0] <= 1
            assert (
                "estimate_order_delivery_time_in_minutes"
                in test_delivery_time.new_analysis_config["covariates"]
            )


def test_invalid_dimension_value(experiment_data, metrics, dimensions, variants):
    """Tests handling of invalid dimension values in hypothesis tests."""
    df, _ = experiment_data
    test_order_value = HypothesisTest(
        metric=metrics["order_value"],
        analysis_type="clustered_ols",
        analysis_config={"cluster_cols": ["customer_id"]},
        dimensions=dimensions,
    )

    # Test with an invalid dimension value
    with pytest.raises(ValueError):
        test_order_value.get_test_results(
            control_variant=variants[0],
            treatment_variant=variants[1],
            variant_col="experiment_group",
            exp_data=df,
            dimension=dimensions[0],
            dimension_value="invalid_value",  # Invalid dimension value
            alpha=0.05,
        )


@pytest.fixture
def aov_metric():
    """Sets up a sample Metric instance for testing."""
    return SimpleMetric(alias="AOV", name="order_value")


class CustomExperimentAnalysis(ClusteredOLSAnalysis):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@pytest.fixture
def custom_analysis_type_mapper():
    """Sets up a custom_analysis_type_mapper for testing."""
    return {"custom_clustered_ols": CustomExperimentAnalysis}


def test_hypothesis_test_with_default_analysis_type(aov_metric, dimensions):
    """Tests that HypothesisTest instantiation checks analysis_type against analysis_mapping if no custom mapper is provided."""
    # Use a valid analysis type from analysis_mapping
    valid_analysis_type = next(iter(analysis_mapping.keys()))

    # Instantiating HypothesisTest with valid analysis_type should not raise an error
    test = HypothesisTest(
        metric=aov_metric,
        analysis_type=valid_analysis_type,
        analysis_config={"cluster_cols": ["customer_id"]},
        dimensions=dimensions,
    )

    # Ensure analysis_type is set correctly
    assert test.analysis_type == valid_analysis_type

    # Try instantiating with an invalid analysis type (not in analysis_mapping)
    invalid_analysis_type = "invalid_analysis_type"
    with pytest.raises(
        ValueError,
        match="Analysis type 'invalid_analysis_type' not found in analysis_mapping",
    ):
        HypothesisTest(
            metric=aov_metric,
            analysis_type=invalid_analysis_type,
            analysis_config={"cluster_cols": ["customer_id"]},
            dimensions=dimensions,
        )


def test_hypothesis_test_with_custom_analysis_type_mapper(
    aov_metric, dimensions, custom_analysis_type_mapper
):
    """Tests that HypothesisTest instantiation checks analysis_type against custom_analysis_type_mapper when provided."""

    # Instantiating HypothesisTest with an analysis_type that exists in the custom mapper should succeed
    test = HypothesisTest(
        metric=aov_metric,
        analysis_type="custom_clustered_ols",
        analysis_config={"cluster_cols": ["customer_id"]},
        dimensions=dimensions,
        custom_analysis_type_mapper=custom_analysis_type_mapper,
    )

    # Ensure analysis_type is set correctly
    assert test.analysis_type == "custom_clustered_ols"

    # Attempt to use an analysis_type not in the custom mapper should raise an error
    with pytest.raises(
        ValueError,
        match="Analysis type 'invalid_custom_type' not found in the provided custom_analysis_type_mapper",
    ):
        HypothesisTest(
            metric=aov_metric,
            analysis_type="invalid_custom_type",
            analysis_config={"cluster_cols": ["customer_id"]},
            dimensions=dimensions,
            custom_analysis_type_mapper=custom_analysis_type_mapper,
        )
