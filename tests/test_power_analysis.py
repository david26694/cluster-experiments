from datetime import date

import pytest
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.power_config import PowerConfig
from cluster_experiments.pre_experiment_covariates import PreExperimentFeaturizer
from cluster_experiments.random_splitter import SwitchbackSplitter

from tests.examples import generate_random_data

N = 10_000


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


def test_power_analysis(df, clusters, experiment_dates):
    sw = SwitchbackSplitter(
        clusters=clusters,
        dates=experiment_dates,
    )

    perturbator = UniformPerturbator(
        average_effect=0.1,
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    featurizer = PreExperimentFeaturizer()

    pw = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        featurizer=featurizer,
        n_simulations=3,
    )

    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1


def test_power_analysis_config(df, clusters, experiment_dates):
    config = PowerConfig(
        clusters=clusters,
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="uniform",
        splitter="switchback",
        dates=experiment_dates,
        n_simulations=4,
    )
    pw = PowerAnalysis.from_config(config)
    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1


def test_power_analysis_dict(df, clusters, experiment_dates):
    config = dict(
        clusters=clusters,
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="uniform",
        splitter="switchback",
        dates=experiment_dates,
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)
    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1
