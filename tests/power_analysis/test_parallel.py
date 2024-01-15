import pytest

from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.power_config import PowerConfig


def test_raise_n_jobs(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_config(config)
    with pytest.raises(ValueError):
        pw.power_analysis(df, average_effect=0.0, n_jobs=0)
    with pytest.raises(ValueError):
        pw.power_analysis(df, average_effect=0.0, n_jobs=-10)


def test_similar_n_jobs(df):
    config = PowerConfig(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=100,
        seed=123,
    )
    pw = PowerAnalysis.from_config(config)
    power = pw.power_analysis(df, average_effect=0.0, n_jobs=1)
    power2 = pw.power_analysis(df, average_effect=0.0, n_jobs=2)
    power3 = pw.power_analysis(df, average_effect=0.0, n_jobs=-1)
    assert abs(power - power2) <= 0.1
    assert abs(power - power3) <= 0.1
