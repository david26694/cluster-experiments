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
    return SwitchbackSplitter(time_col="time", switch_frequency="1D")


@pytest.fixture
def switchback_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "uniform",
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
    return BalancedSwitchbackSplitter(time_col="time", switch_frequency="1D")


@pytest.fixture
def balanced_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "uniform",
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
    )


@pytest.fixture
def stratified_switchback_splitter_config():
    config = {
        "time_col": "time",
        "switch_frequency": "1D",
        "perturbator": "uniform",
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
