import pytest

from cluster_experiments import (
    NonClusteredSplitter,
    OLSAnalysis,
    PowerAnalysis,
    UniformPerturbator,
)
from tests.examples import generate_non_clustered_data


@pytest.fixture
def df():
    return generate_non_clustered_data(
        N=1000,
        n_users=100,
    )


@pytest.fixture
def perturbator():
    return UniformPerturbator()


@pytest.fixture
def ols():
    return OLSAnalysis()


@pytest.fixture
def binary_hypothesis_power(perturbator, ols):
    splitter = NonClusteredSplitter(
        treatments=["A", "B"],
        treatment_col="treatment",
    )
    return PowerAnalysis(
        splitter=splitter,
        perturbator=perturbator,
        analysis=ols,
    )


@pytest.fixture
def multivariate_hypothesis_power(perturbator, ols):
    splitter = NonClusteredSplitter(
        treatments=["A", "B", "C", "D", "E", "F", "G"],
        treatment_col="treatment",
    )
    return PowerAnalysis(
        splitter=splitter,
        perturbator=perturbator,
        analysis=ols,
    )


@pytest.fixture
def binary_hypothesis_power_config():
    config = {
        "analysis": "ols_non_clustered",
        "perturbator": "uniform",
        "splitter": "non_clustered",
        "n_simulations": 50,
    }
    return PowerAnalysis.from_dict(config)


@pytest.fixture
def multivariate_hypothesis_power_config():
    config = {
        "analysis": "ols_non_clustered",
        "perturbator": "uniform",
        "splitter": "non_clustered",
        "n_simulations": 50,
        "treatments": ["A", "B", "C", "D", "E", "F", "G"],
    }
    return PowerAnalysis.from_dict(config)


def test_higher_power_analysis(
    multivariate_hypothesis_power,
    binary_hypothesis_power,
    df,
):
    power_multi = multivariate_hypothesis_power.power_analysis(df, average_effect=0.1)
    power_binary = binary_hypothesis_power.power_analysis(df, average_effect=0.1)
    assert power_multi < power_binary, f"{power_multi = } > {power_binary = }"


def test_higher_power_analysis_config(
    multivariate_hypothesis_power_config,
    binary_hypothesis_power_config,
    df,
):
    power_multi = multivariate_hypothesis_power_config.power_analysis(
        df, average_effect=0.1
    )
    power_binary = binary_hypothesis_power_config.power_analysis(df, average_effect=0.1)
    assert power_multi < power_binary, f"{power_multi = } > {power_binary = }"


def test_raise_if_control_not_in_treatments(
    perturbator,
    ols,
):
    with pytest.raises(AssertionError):
        splitter = NonClusteredSplitter(
            treatments=["a", "B"],
            treatment_col="treatment",
            splitter_weights=[0.5, 0.5],
        )
        PowerAnalysis(
            splitter=splitter,
            perturbator=perturbator,
            analysis=ols,
        )
    with pytest.raises(AssertionError):
        splitter = NonClusteredSplitter(
            treatment_col="treatment",
            splitter_weights=[0.5, 0.5],
        )
        PowerAnalysis(
            splitter=splitter, perturbator=perturbator, analysis=ols, control="X"
        )
