import pytest

from cluster_experiments.inference.variant import Variant


def test_variant_initialization():
    """Test Variant initialization with valid inputs."""
    variant = Variant(name="Test Variant", is_control=True)
    assert variant.name == "Test Variant"
    assert variant.is_control is True


def test_variant_name_type():
    """Test that Variant raises TypeError if name is not a string."""
    with pytest.raises(TypeError, match="Variant name must be a string"):
        Variant(name=123, is_control=True)  # Name should be a string


def test_variant_is_control_type():
    """Test that Variant raises TypeError if is_control is not a boolean."""
    with pytest.raises(TypeError, match="Variant is_control must be a boolean"):
        Variant(name="Test Variant", is_control="yes")  # is_control should be a boolean


def test_variant_is_control_default_behavior():
    """Test Variant behavior when is_control is set to False."""
    variant = Variant(name="Variant B", is_control=False)
    assert variant.name == "Variant B"
    assert variant.is_control is False
