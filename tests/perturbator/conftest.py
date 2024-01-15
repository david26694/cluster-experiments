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


@pytest.fixture
def continuous_df():
    return pd.DataFrame(
        {
            "target": [0.5, 0.5, 0.5, 0.5],
            "treatment": ["A", "B", "B", "A"],
        }
    )


@pytest.fixture
def generate_clustered_data() -> pd.DataFrame:
    analysis_df = pd.DataFrame(
        {
            "country_code": ["ES"] * 4 + ["IT"] * 4 + ["PL"] * 4 + ["RO"] * 4,
            "city_code": ["BCN", "BCN", "MAD", "BCN"]
            + ["NAP"] * 4
            + ["WAW"] * 4
            + ["BUC"] * 4,
            "user_id": [1, 1, 2, 1, 3, 4, 5, 6, 7, 8, 8, 8, 9, 9, 9, 10],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"] * 4,
            "treatment": [
                "A",
                "A",
                "B",
                "A",
                "B",
                "B",
                "A",
                "B",
                "B",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "A",
            ],  # Randomization is done at user level, so same user will always have same treatment
            "target": [0.01] * 15 + [0.1],
        }
    )
    return analysis_df
