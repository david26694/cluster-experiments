import numpy as np

from cluster_experiments.power_analysis import PowerAnalysis


def get_config(perturbator: str) -> dict:
    return {
        "cluster_cols": ["cluster"],
        "analysis": "ols_clustered",
        "splitter": "clustered",
        "n_simulations": 15,
        "seed": 123,
        "perturbator": perturbator,
    }


def test_power_analysis_constant_perturbator_seed(df):
    config_dict = get_config("constant")

    powers = []
    for _ in range(10):
        pw = PowerAnalysis.from_dict(config_dict)
        powers.append(pw.power_analysis(df, average_effect=10))

    assert np.var(np.asarray(powers)) == 0


def test_power_analysis_binary_perturbator_seed(df_binary):
    config_dict = get_config("binary")

    powers = []
    for _ in range(10):
        pw = PowerAnalysis.from_dict(config_dict)
        powers.append(pw.power_analysis(df_binary, average_effect=0.08))

    assert np.var(np.asarray(powers)) == 0
