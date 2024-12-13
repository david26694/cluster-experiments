import pytest

from ab_lab.inference.dimension import DefaultDimension, Dimension


def test_dimension_initialization():
    """Test Dimension initialization with valid inputs."""
    dim = Dimension(name="Country", values=["US", "CA", "UK"])
    assert dim.name == "Country"
    assert dim.values == ["US", "CA", "UK"]


def test_dimension_name_type():
    """Test that Dimension raises TypeError if name is not a string."""
    with pytest.raises(TypeError, match="Dimension name must be a string"):
        Dimension(name=123, values=["US", "CA", "UK"])  # Name should be a string


def test_dimension_values_type():
    """Test that Dimension raises TypeError if values is not a list of strings."""
    # Values should be a list
    with pytest.raises(TypeError, match="Dimension values must be a list of strings"):
        Dimension(name="Country", values="US, CA, UK")  # Should be a list of strings

    # Values should be a list of strings
    with pytest.raises(TypeError, match="Dimension values must be a list of strings"):
        Dimension(
            name="Country", values=["US", 123, "UK"]
        )  # All elements should be strings


def test_dimension_iterate_dimension_values():
    """Test Dimension iterate_dimension_values method to ensure unique values are returned."""
    dim = Dimension(name="Country", values=["US", "CA", "US", "UK", "CA"])
    unique_values = list(dim.iterate_dimension_values())
    assert unique_values == ["US", "CA", "UK"]  # Ensures unique, ordered values


def test_default_dimension_initialization():
    """Test DefaultDimension initialization."""
    default_dim = DefaultDimension()
    assert default_dim.name == "__total_dimension"
    assert default_dim.values == ["total"]


def test_default_dimension_iterate_dimension_values():
    """Test that DefaultDimension's iterate_dimension_values yields 'total'."""
    default_dim = DefaultDimension()
    values = list(default_dim.iterate_dimension_values())
    assert values == ["total"]
