import numpy as np
import pytest
from cluster_experiments.cupac import TargetAggregation
from cluster_experiments.experiment_analysis import OLSAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import NonClusteredSplitter
from sklearn.model_selection import train_test_split

from tests.examples import generate_non_clustered_data

N = 1_000
n_users = 100


@pytest.fixture
def df():
    return generate_non_clustered_data(N, n_users)


@pytest.fixture
def df_feats():
    df = generate_non_clustered_data(N, n_users)
    df["x1"] = np.random.normal(0, 1, N)
    df["x2"] = np.random.normal(0, 1, N)
    return df


@pytest.fixture
def cupac_power_analysis():
    sw = NonClusteredSplitter()

    perturbator = UniformPerturbator(
        average_effect=0.1,
    )

    analysis = OLSAnalysis(
        covariates=["estimate_target"],
    )

    target_agg = TargetAggregation(
        agg_col="user",
    )

    return PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis,
        cupac_model=target_agg,
        n_simulations=3,
    )


@pytest.fixture
def cupac_from_config():
    return PowerAnalysis.from_dict(
        dict(
            analysis="ols_non_clustered",
            perturbator="uniform",
            splitter="non_clustered",
            cupac_model="mean_cupac_model",
            average_effect=0.1,
            n_simulations=4,
            covariates=["estimate_target"],
            agg_col="user",
        )
    )


def test_power_analysis(cupac_power_analysis, df):
    pre_df, df = train_test_split(df)
    power = cupac_power_analysis.power_analysis(df, pre_df)
    assert power >= 0
    assert power <= 1


def test_power_analysis_config(cupac_from_config, df):
    pre_df, df = train_test_split(df)
    power = cupac_from_config.power_analysis(df, pre_df)
    assert power >= 0
    assert power <= 1
