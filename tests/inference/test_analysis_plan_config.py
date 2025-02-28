import numpy as np
import pandas as pd
import pytest

from cluster_experiments.inference.analysis_plan import AnalysisPlan
from cluster_experiments.inference.dimension import Dimension
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import SimpleMetric
from cluster_experiments.inference.variant import Variant


@pytest.fixture
def experiment_data():
    N = 1_000
    return pd.DataFrame(
        {
            "order_value": np.random.normal(100, 10, size=N),
            "delivery_time": np.random.normal(10, 1, size=N),
            "experiment_group": np.random.choice(["control", "treatment"], size=N),
            "city": np.random.choice(["NYC", "LA"], size=N),
            "customer_id": np.random.randint(1, 100, size=N),
            "customer_age": np.random.randint(20, 60, size=N),
        }
    )


def test_from_metrics_dict():
    d = {
        "metrics": [{"alias": "AOV", "name": "order_value"}],
        "variants": [
            {"name": "control", "is_control": True},
            {"name": "treatment_1", "is_control": False},
        ],
        "variant_col": "experiment_group",
        "alpha": 0.05,
        "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
        "analysis_type": "clustered_ols",
        "analysis_config": {"cluster_cols": ["customer_id"]},
    }
    plan = AnalysisPlan.from_metrics_dict(d)
    assert isinstance(plan, AnalysisPlan)
    assert len(plan.tests) == 1
    assert isinstance(plan.tests[0], HypothesisTest)
    assert plan.variant_col == "experiment_group"
    assert plan.alpha == 0.05
    assert len(plan.variants) == 2


def test_analyze_from_metrics_dict(experiment_data):
    # given
    plan = AnalysisPlan.from_metrics_dict(
        {
            "metrics": [
                {"alias": "AOV", "name": "order_value"},
                {"alias": "delivery_time", "name": "delivery_time"},
            ],
            "variants": [
                {"name": "control", "is_control": True},
                {"name": "treatment", "is_control": False},
            ],
            "variant_col": "experiment_group",
            "alpha": 0.05,
            "dimensions": [
                {"name": "city", "values": ["NYC", "LA"]},
            ],
            "analysis_type": "clustered_ols",
            "analysis_config": {"cluster_cols": ["customer_id"]},
        }
    )

    # when
    results = plan.analyze(experiment_data)
    results_df = results.to_dataframe()

    # then
    assert (
        len(results_df) == 6
    ), "There should be 6 rows in the results DataFrame, 2 metrics x 3 dimension values"
    assert set(results_df["metric_alias"]) == {
        "AOV",
        "delivery_time",
    }, "The metric aliases should be present in the DataFrame"
    assert set(results_df["dimension_value"]) == {
        "total",
        "" "NYC",
        "LA",
    }, "The dimension values should be present in the DataFrame"


def test_from_dict_config():
    # ensures that we get the same object when creating from a dict or a config
    # given
    d = {
        "tests": [
            {
                "metric": {"alias": "AOV", "name": "order_value"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
            {
                "metric": {"alias": "delivery_time", "name": "delivery_time"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
        ],
        "variants": [
            {"name": "control", "is_control": True},
            {"name": "treatment", "is_control": False},
        ],
        "variant_col": "experiment_group",
        "alpha": 0.05,
    }
    plan = AnalysisPlan(
        variants=[
            Variant(name="control", is_control=True),
            Variant(name="treatment", is_control=False),
        ],
        variant_col="experiment_group",
        alpha=0.05,
        tests=[
            HypothesisTest(
                metric=SimpleMetric(alias="AOV", name="order_value"),
                analysis_type="clustered_ols",
                analysis_config={"cluster_cols": ["customer_id"]},
                dimensions=[Dimension(name="city", values=["NYC", "LA"])],
            ),
            HypothesisTest(
                metric=SimpleMetric(alias="delivery_time", name="delivery_time"),
                analysis_type="clustered_ols",
                analysis_config={"cluster_cols": ["customer_id"]},
                dimensions=[Dimension(name="city", values=["NYC", "LA"])],
            ),
        ],
    )

    # when
    plan_from_config = AnalysisPlan.from_dict(d)

    # then
    assert plan.variant_col == plan_from_config.variant_col
    assert plan.alpha == plan_from_config.alpha
    for variant in plan.variants:
        assert variant in plan_from_config.variants


def test_from_dict():
    # given
    d = {
        "tests": [
            {
                "metric": {"alias": "AOV", "name": "order_value"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
            {
                "metric": {"alias": "DT", "name": "delivery_time"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
        ],
        "variants": [
            {"name": "control", "is_control": True},
            {"name": "treatment_1", "is_control": False},
        ],
        "variant_col": "experiment_group",
        "alpha": 0.05,
    }

    # when
    plan = AnalysisPlan.from_dict(d)

    # then
    assert isinstance(plan, AnalysisPlan)
    assert len(plan.tests) == 2
    assert isinstance(plan.tests[0], HypothesisTest)
    assert plan.variant_col == "experiment_group"
    assert plan.alpha == 0.05
    assert len(plan.variants) == 2
    assert plan.tests[1].metric.alias == "DT"


def test_analyze_from_dict(experiment_data):
    # given
    d = {
        "tests": [
            {
                "metric": {"alias": "AOV", "name": "order_value"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
            {
                "metric": {"alias": "DT", "name": "delivery_time"},
                "analysis_type": "clustered_ols",
                "analysis_config": {"cluster_cols": ["customer_id"]},
                "dimensions": [{"name": "city", "values": ["NYC", "LA"]}],
            },
        ],
        "variants": [
            {"name": "control", "is_control": True},
            {"name": "treatment_1", "is_control": False},
        ],
        "variant_col": "experiment_group",
        "alpha": 0.05,
    }
    plan = AnalysisPlan.from_dict(d)

    # when
    results = plan.analyze(experiment_data)
    results_df = results.to_dataframe()

    # then
    assert (
        len(results_df) == 6
    ), "There should be 6 rows in the results DataFrame, 2 metrics x 3 dimension values"
    assert set(results_df["metric_alias"]) == {
        "AOV",
        "DT",
    }, "The metric aliases should be present in the DataFrame"
    assert set(results_df["dimension_value"]) == {
        "total",
        "" "NYC",
        "LA",
    }, "The dimension values should be present in the DataFrame"
