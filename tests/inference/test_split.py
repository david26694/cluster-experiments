import pytest

from cluster_experiments.inference.split import DefaultSplit, Split


def test_split_initialization():
    """Test Split initialization with valid inputs."""
    # Using 'Status' as a split example
    s = Split(name="Status", values=["Prime", "Non-Prime"])
    assert s.name == "Status"
    assert s.values == ["Prime", "Non-Prime"]


def test_split_name_type():
    """Test that Split raises TypeError if name is not a string."""
    with pytest.raises(TypeError, match="Dimension name must be a string"):
        Split(name=123, values=["A", "B"])


def test_split_values_type():
    """Test that Split raises TypeError if values is not a list of strings."""
    # Values should be a list
    with pytest.raises(TypeError, match="Dimension values must be a list of strings"):
        Split(name="Status", values="Prime, Non-Prime")

    # Values should be a list of strings
    with pytest.raises(TypeError, match="Dimension values must be a list of strings"):
        Split(name="Status", values=["Prime", 123])


def test_split_iterate_values():
    """Test Split iterate_dimension_values method to ensure unique values are returned."""
    # Same logic as your Country example, but with Split
    s = Split(name="Status", values=["Prime", "Non-Prime", "Prime", "Other"])
    unique_values = list(s.iterate_dimension_values())
    assert unique_values == ["Prime", "Non-Prime", "Other"]


def test_default_split_initialization():
    """Test DefaultSplit initialization."""
    default_s = DefaultSplit()
    # This checks our specific implementation of DefaultSplit
    assert default_s.name == "__total_split"
    assert default_s.values == ["total"]


def test_default_split_iterate_dimension_values():
    """Test that DefaultSplit's iterate_dimension_values yields 'total'."""
    default_s = DefaultSplit()
    values = list(default_s.iterate_dimension_values())
    assert values == ["total"]


def test_default_split_str():
    """Test the __str__ method of DefaultSplit."""
    default_s = DefaultSplit()
    assert str(default_s) == "DefaultSplit(total)"
