from datetime import date

import pytest
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.pre_experiment_covariates import Aggregator
from cluster_experiments.random_splitter import SwitchbackSplitter

from tests.examples import generate_random_data


@pytest.mark.unit
def test_power_analysis():
    clusters = [f"Cluster {i}" for i in range(100)]
    dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
    experiment_dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]
    N = 10_000
    df = generate_random_data(clusters, dates, N)
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

    aggregator = Aggregator()

    pw = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        aggregator=aggregator,
        n_simulations=3,
    )

    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1
