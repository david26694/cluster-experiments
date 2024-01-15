from datetime import date, timedelta

import numpy as np
import pytest

from cluster_experiments.cupac import TargetAggregation
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import (
    ClusteredSplitter,
    StratifiedSwitchbackSplitter,
)
from tests.utils import generate_random_data

N = 1_000


@pytest.fixture
def clusters():
    return [f"Cluster {i}" for i in range(100)]


@pytest.fixture
def dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]


@pytest.fixture
def experiment_dates():
    return [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]


@pytest.fixture
def df(clusters, dates):
    return generate_random_data(clusters, dates, N)


@pytest.fixture
def df_feats(clusters, dates):
    df = generate_random_data(clusters, dates, N)
    df["x1"] = np.random.normal(0, 1, N)
    df["x2"] = np.random.normal(0, 1, N)
    return df


@pytest.fixture
def df_binary(clusters, dates):
    return generate_random_data(clusters, dates, N, target="binary")


@pytest.fixture
def perturbator():
    return ConstantPerturbator(average_effect=0.1)


@pytest.fixture
def analysis_gee_vainilla():
    return GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )


@pytest.fixture
def analysis_gee():
    return GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
        covariates=["estimate_target"],
    )


@pytest.fixture
def cupac_power_analysis(perturbator, analysis_gee):
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    target_agg = TargetAggregation(
        agg_col="cluster",
    )

    return PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis_gee,
        cupac_model=target_agg,
        n_simulations=3,
    )


@pytest.fixture
def switchback_power_analysis(perturbator, analysis_gee_vainilla):
    sw = StratifiedSwitchbackSplitter(
        time_col="date",
        switch_frequency="1D",
        strata_cols=["cluster"],
        cluster_cols=["cluster", "date"],
    )

    return PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis_gee_vainilla,
        n_simulations=3,
    )


@pytest.fixture
def switchback_power_analysis_hourly(perturbator, analysis_gee_vainilla):
    sw = StratifiedSwitchbackSplitter(
        time_col="date",
        switch_frequency="1H",
        strata_cols=["cluster"],
        cluster_cols=["cluster", "date"],
    )

    return PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis_gee_vainilla,
        n_simulations=3,
    )


@pytest.fixture
def switchback_washover():
    return PowerAnalysis.from_dict(
        {
            "time_col": "date",
            "switch_frequency": "1D",
            "perturbator": "constant",
            "analysis": "ols_clustered",
            "splitter": "switchback_balance",
            "cluster_cols": ["cluster", "date"],
            "strata_cols": ["cluster"],
            "washover": "constant_washover",
            "washover_time_delta": timedelta(hours=2),
        }
    )
