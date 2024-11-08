import numpy as np
import pandas as pd
import pytest

from cluster_experiments.inference.analysis_plan import AnalysisPlan
from cluster_experiments.inference.analysis_results import AnalysisPlanResults
from cluster_experiments.inference.dimension import Dimension
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import SimpleMetric
from cluster_experiments.inference.variant import Variant

# Set up constants for the data
NUM_ORDERS = 10000
NUM_CUSTOMERS = 3000
EXPERIMENT_GROUPS = ["control", "treatment_1", "treatment_2"]
GROUP_SIZE = NUM_CUSTOMERS // len(EXPERIMENT_GROUPS)


@pytest.fixture
def exp_data():
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

    # Build experimental data
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

    return df


@pytest.fixture
def pre_exp_data(exp_data):
    """Sample pre-experiment data for testing."""
    pre_exp_df = exp_data.assign(
        order_value=lambda df: df["order_value"]
        + np.random.normal(loc=0, scale=1, size=NUM_ORDERS),
        order_delivery_time_in_minutes=lambda df: df["order_delivery_time_in_minutes"]
        + np.random.normal(loc=0, scale=2, size=NUM_ORDERS),
    ).sample(int(NUM_ORDERS / 3))

    return pre_exp_df


@pytest.fixture
def sample_variants():
    return [
        Variant("control", is_control=True),
        Variant("treatment_1", is_control=False),
        Variant("treatment_2", is_control=False),
    ]


@pytest.fixture
def sample_tests():
    metric = SimpleMetric(alias="AOV", name="order_value")
    dimension = Dimension(name="order_city_code", values=["1", "2"])
    test = HypothesisTest(
        metric=metric,
        analysis_type="clustered_ols",
        analysis_config={"cluster_cols": ["customer_id"]},
        dimensions=[dimension],
    )
    return [test]


def test_analysis_plan_initialization(sample_tests, sample_variants):
    """Tests initialization of AnalysisPlan with valid inputs."""
    plan = AnalysisPlan(tests=sample_tests, variants=sample_variants)
    assert plan.tests == sample_tests
    assert plan.variants == sample_variants
    assert plan.variant_col == "treatment"
    assert plan.alpha == 0.05


def test_analysis_plan_invalid_inputs(sample_tests):
    """Tests invalid inputs for AnalysisPlan initialization."""
    with pytest.raises(TypeError):
        AnalysisPlan(tests="not_a_list", variants=[Variant("control", is_control=True)])
    with pytest.raises(TypeError):
        AnalysisPlan(tests=sample_tests, variants="not_a_list")
    with pytest.raises(ValueError):
        AnalysisPlan(tests=[], variants=[Variant("control", is_control=True)])
    with pytest.raises(ValueError):
        AnalysisPlan(tests=sample_tests, variants=[])


def test_analyze_method(sample_tests, sample_variants, exp_data, pre_exp_data):
    """Tests the analyze method with valid inputs."""
    plan = AnalysisPlan(
        tests=sample_tests, variants=sample_variants, variant_col="experiment_group"
    )
    results = plan.analyze(exp_data=exp_data, pre_exp_data=pre_exp_data)
    assert isinstance(results, AnalysisPlanResults)


def test_analyze_data_validation(sample_tests, sample_variants):
    """Tests data validation in the analyze method."""
    plan = AnalysisPlan(tests=sample_tests, variants=sample_variants)

    with pytest.raises(ValueError, match="exp_data must be a pandas DataFrame"):
        plan.analyze(exp_data="not_a_dataframe")

    with pytest.raises(ValueError, match="exp_data cannot be empty"):
        plan.analyze(exp_data=pd.DataFrame())

    with pytest.raises(ValueError, match="pre_exp_data must be a pandas DataFrame"):
        plan.analyze(
            exp_data=pd.DataFrame({"dummy": [1]}), pre_exp_data="not_a_dataframe"
        )

    with pytest.raises(ValueError, match="pre_exp_data cannot be empty"):
        plan.analyze(exp_data=pd.DataFrame({"dummy": [1]}), pre_exp_data=pd.DataFrame())


def test_control_variant_property(sample_tests, sample_variants):
    """Tests that the control_variant property returns the correct control variant."""
    plan = AnalysisPlan(tests=sample_tests, variants=sample_variants)
    assert plan.control_variant.name == "control"

    # Test when no control variant is present
    no_control_variants = [Variant("treatment_1", is_control=False)]
    plan_no_control = AnalysisPlan(tests=sample_tests, variants=no_control_variants)
    with pytest.raises(ValueError, match="No control variant found"):
        _ = plan_no_control.control_variant


def test_treatment_variants_property(sample_tests, sample_variants):
    """Tests that the treatment_variants property returns the correct treatment variants."""
    plan = AnalysisPlan(tests=sample_tests, variants=sample_variants)
    treatments = plan.treatment_variants
    assert len(treatments) == 2
    assert treatments[0].name == "treatment_1"
    assert treatments[1].name == "treatment_2"

    # Test when no treatment variants are present
    only_control_variant = [Variant("control", is_control=True)]
    plan_only_control = AnalysisPlan(tests=sample_tests, variants=only_control_variant)
    with pytest.raises(ValueError, match="No treatment variants found"):
        _ = plan_only_control.treatment_variants


def test_from_metrics_classmethod(sample_variants):
    """Tests the from_metrics class method for creating an AnalysisPlan instance."""
    metric = SimpleMetric(alias="AOV", name="order_value")
    metrics = [metric]
    dimensions = [Dimension(name="order_city_code", values=["1", "2"])]

    plan = AnalysisPlan.from_metrics(
        metrics=metrics,
        variants=sample_variants,
        variant_col="experiment_group",
        alpha=0.05,
        dimensions=dimensions,
        analysis_type="clustered_ols",
        analysis_config={"cluster_cols": ["customer_id"]},
    )

    assert isinstance(plan, AnalysisPlan)
    assert len(plan.tests) == 1
    assert isinstance(plan.tests[0], HypothesisTest)
    assert plan.tests[0].metric == metric
    assert plan.variant_col == "experiment_group"
    assert plan.alpha == 0.05
