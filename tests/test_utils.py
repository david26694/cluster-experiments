import pytest

from cluster_experiments.utils import _get_mapping_key


def test_get_mapping_key():
    with pytest.raises(KeyError):
        mapping = {"a": 1, "b": 2}
        _get_mapping_key(mapping, "c")
