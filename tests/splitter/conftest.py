import random

import numpy as np
import pandas as pd
import pytest

from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import (
    BalancedSwitchbackSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
)


@pytest.fixture
def switchback_splitter():
    return SwitchbackSplitter(
        time_col="time", switch_frequency="1D", cluster_cols=["time"]
    )


@pytest.fixture
def switchback_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "constant",
        "analysis": "ols_clustered",
        "splitter": "switchback",
        "cluster_cols": ["time"],
    }
    return PowerAnalysis.from_dict(config).splitter


switchback_splitter_parametrize = pytest.mark.parametrize(
    "splitter",
    [
        "switchback_splitter",
        "switchback_splitter_config",
    ],
)


@pytest.fixture
def balanced_splitter():
    return BalancedSwitchbackSplitter(
        time_col="time", switch_frequency="1D", cluster_cols=["time"]
    )


@pytest.fixture
def balanced_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "constant",
        "analysis": "ols_clustered",
        "splitter": "switchback_balance",
        "cluster_cols": ["time"],
    }
    return PowerAnalysis.from_dict(config).splitter


balanced_splitter_parametrize = pytest.mark.parametrize(
    "splitter",
    [
        "balanced_splitter",
        "balanced_splitter_config",
    ],
)


@pytest.fixture
def stratified_switchback_splitter():
    return StratifiedSwitchbackSplitter(
        time_col="time",
        switch_frequency="1D",
        strata_cols=["day_of_week"],
        cluster_cols=["time"],
    )


@pytest.fixture
def stratified_switchback_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "constant",
        "analysis": "ols_clustered",
        "splitter": "switchback_stratified",
        "cluster_cols": ["time"],
        "strata_cols": ["day_of_week"],
    }
    return PowerAnalysis.from_dict(config).splitter


stratified_splitter_parametrize = pytest.mark.parametrize(
    "splitter",
    [
        "stratified_switchback_splitter",
        "stratified_switchback_splitter_config",
    ],
)


@pytest.fixture
def date_df():
    return pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", "2020-01-10", freq="1D"),
            "y": np.random.randn(10),
        }
    )


@pytest.fixture
def biweekly_df():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", "2020-01-14", freq="1D"),
            "y": np.random.randn(14),
        }
    ).assign(
        day_of_week=lambda df: df["time"].dt.day_name(),
    )
    return pd.concat([df.assign(cluster=f"Cluster {i}") for i in range(10)])


@pytest.fixture
def washover_df():
    # Define data with random dates
    df_raw = pd.DataFrame(
        {
            "time": pd.date_range("2021-01-01", "2021-01-02", freq="1min")[
                np.random.randint(24 * 60, size=7 * 24 * 60)
            ],
            "y": np.random.randn(7 * 24 * 60),
        }
    ).assign(
        day_of_week=lambda df: df.time.dt.dayofweek,
        hour_of_day=lambda df: df.time.dt.hour,
    )
    df = pd.concat([df_raw.assign(city=city) for city in ("TGN", "NYC", "LON", "REU")])
    return df


@pytest.fixture
def washover_base_df():
    df = pd.DataFrame(
        {
            "original___time": [
                pd.to_datetime("2022-01-01 00:20:00"),
                pd.to_datetime("2022-01-01 00:31:00"),
                pd.to_datetime("2022-01-01 01:14:00"),
                pd.to_datetime("2022-01-01 01:31:00"),
            ],
            "treatment": ["A", "A", "B", "B"],
            "city": ["TGN"] * 4,
        }
    ).assign(time=lambda x: x["original___time"].dt.floor("1h"))
    return df


@pytest.fixture
def washover_df_no_switch():
    df = pd.DataFrame(
        {
            "original___time": [
                pd.to_datetime("2022-01-01 00:20:00"),
                pd.to_datetime("2022-01-01 00:31:00"),
                pd.to_datetime("2022-01-01 01:14:00"),
                pd.to_datetime("2022-01-01 01:31:00"),
                pd.to_datetime("2022-01-01 02:01:00"),
                pd.to_datetime("2022-01-01 02:31:00"),
            ],
            "treatment": ["A", "A", "B", "B", "B", "B"],
            "city": ["TGN"] * 6,
        }
    ).assign(time=lambda x: x["original___time"].dt.floor("1h"))
    return df


@pytest.fixture
def washover_df_multi_city():
    df = pd.DataFrame(
        {
            "original___time": [
                pd.to_datetime("2022-01-01 00:20:00"),
                pd.to_datetime("2022-01-01 00:31:00"),
                pd.to_datetime("2022-01-01 01:14:00"),
                pd.to_datetime("2022-01-01 01:31:00"),
                pd.to_datetime("2022-01-01 02:01:00"),
                pd.to_datetime("2022-01-01 02:31:00"),
            ]
            * 2,
            "treatment": ["A", "A", "B", "B", "B", "B"]
            + ["A", "A", "A", "A", "B", "B"],
            "city": ["TGN"] * 6 + ["BCN"] * 6,
        }
    ).assign(time=lambda x: x["original___time"].dt.floor("1h"))
    return df


@pytest.fixture
def washover_split_df(n):
    # Return
    return pd.DataFrame(
        {
            # Random time each minute in 2022-01-01, length 1000
            "time": pd.date_range("2022-01-01", "2022-01-02", freq="1min")[
                np.random.randint(24 * 60, size=n)
            ],
            "city": random.choices(["TGN", "NYC", "LON", "REU"], k=n),
        }
    )
