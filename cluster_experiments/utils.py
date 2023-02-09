def _original_time_column(time_col: str) -> str:
    """
    Usage:
    ```python
    from cluster_experiments.utils import _original_time_column

    assert _original_time_column("hola") == "original___hola"
    ```
    """
    return f"original___{time_col}"
