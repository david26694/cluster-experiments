import pandas as pd
import pytest


@pytest.fixture
def analysis_df():
    return pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "treatment": ["A", "B", "B", "A"],
            "cluster": ["Cluster 1", "Cluster 1", "Cluster 1", "Cluster 1"],
            "date": ["2022-01-01", "2022-01-01", "2022-01-01", "2022-01-01"],
        }
    )
