import pandas as pd

binary_df = pd.DataFrame(
    {
        "target": [0, 1, 0, 1],
        "treatment": ["A", "B", "B", "A"],
    }
)


continuous_df = pd.DataFrame(
    {
        "target": [0.5, 0.5, 0.5, 0.5],
        "treatment": ["A", "B", "B", "A"],
    }
)
