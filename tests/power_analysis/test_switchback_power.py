import numpy as np
import pandas as pd

from cluster_experiments.power_analysis import PowerAnalysis


def test_switchback(switchback_power_analysis, df):
    power = switchback_power_analysis.power_analysis(
        df,
        average_effect=0.0,
        verbose=True,
    )
    assert power >= 0
    assert power <= 1


def test_switchback_hour(switchback_power_analysis, df):
    # Random dates in 2022-06-26 00:00:00 - 2022-06-26 23:00:00
    df["date"] = pd.to_datetime(
        np.random.randint(
            1624646400,
            1624732800,
            size=len(df),
        ),
        unit="s",
    )
    power = switchback_power_analysis.power_analysis(
        df,
        average_effect=0.0,
        verbose=True,
    )
    assert power >= 0
    assert power <= 1


def test_switchback_strata():

    # Define bihourly switchback splitter
    config = {
        "time_col": "time",
        "switch_frequency": "30min",
        "perturbator": "uniform",
        "analysis": "ols_clustered",
        "splitter": "switchback_stratified",
        "cluster_cols": ["time", "city"],
        "strata_cols": ["day_of_week", "hour_of_day", "city"],
        "target_col": "y",
        "n_simulations": 3,
    }

    power = PowerAnalysis.from_dict(config)
    np.random.seed(42)
    df_raw = pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", "2021-01-10 23:59", freq="1T"),
            "y": np.random.randn(10 * 24 * 60),
        }
    ).assign(
        day_of_week=lambda df: df.time.dt.dayofweek,
        hour_of_day=lambda df: df.time.dt.hour,
    )
    df = pd.concat([df_raw.assign(city=city) for city in ("TGN", "NYC", "LON")])
    pw = power.power_analysis(df, average_effect=0.1)
    assert pw >= 0
    assert pw <= 1
