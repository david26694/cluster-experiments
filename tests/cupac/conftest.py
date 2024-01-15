import pandas as pd
import pytest


@pytest.fixture
def binary_df():
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "treatment": ["A", "B", "B", "A"],
        }
    )
