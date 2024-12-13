import pandas as pd
import pytest

from ab_lab.inference.metric import Metric, RatioMetric, SimpleMetric

# Sample DataFrame for testing
sample_data = pd.DataFrame(
    {
        "salary": [50000, 60000, 70000],
        "numerator": [150000, 180000, 210000],
        "denominator": [3, 3, 3],
    }
)


def test_metric_abstract_instantiation():
    """Test that Metric cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Metric("test_metric")


def test_metric_alias_type():
    """Test that Metric raises TypeError if alias is not a string."""
    with pytest.raises(TypeError):
        SimpleMetric(123, "salary")  # Alias should be a string


def test_simple_metric_initialization():
    """Test SimpleMetric initialization and target column."""
    metric = SimpleMetric("test_metric", "salary")
    assert metric.alias == "test_metric"
    assert metric.target_column == "salary"


def test_simple_metric_name_type():
    """Test that SimpleMetric raises TypeError if name is not a string."""
    with pytest.raises(TypeError):
        SimpleMetric("test_metric", 123)  # Name should be a string


def test_simple_metric_get_mean():
    """Test SimpleMetric get_mean() calculation."""
    metric = SimpleMetric("test_metric", "salary")
    mean_value = metric.get_mean(sample_data)
    assert mean_value == 60000  # Mean of [50000, 60000, 70000]


def test_ratio_metric_initialization():
    """Test RatioMetric initialization and target column."""
    metric = RatioMetric("test_ratio_metric", "numerator", "denominator")
    assert metric.alias == "test_ratio_metric"
    assert metric.target_column == "numerator"


def test_ratio_metric_names_type():
    """Test that RatioMetric raises TypeError if numerator or denominator are not strings."""
    with pytest.raises(TypeError):
        RatioMetric(
            "test_ratio_metric", "numerator", 123
        )  # Denominator should be a string
    with pytest.raises(TypeError):
        RatioMetric(
            "test_ratio_metric", 123, "denominator"
        )  # Numerator should be a string


def test_ratio_metric_get_mean():
    """Test RatioMetric get_mean() calculation."""
    metric = RatioMetric("test_ratio_metric", "numerator", "denominator")
    mean_value = metric.get_mean(sample_data)
    expected_value = sample_data["numerator"].mean() / sample_data["denominator"].mean()
    assert mean_value == expected_value
