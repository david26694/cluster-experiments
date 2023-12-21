from enum import Enum


def _original_time_column(time_col: str) -> str:
    """
    Usage:
    ```python
    from cluster_experiments.utils import _original_time_column

    assert _original_time_column("hola") == "original___hola"
    ```
    """
    return f"original___{time_col}"


def _get_mapping_key(mapping, key):
    try:
        return mapping[key]
    except KeyError:
        raise KeyError(
            f"Could not find {key = } in mapping. All options are the following: {list(mapping.keys())}"
        )


class HypothesisEntries(Enum):
    TWO_SIDED = "two-sided"
    LESS = "less"
    GREATER = "greater"
