from cluster_experiments.power_analysis import NormalPowerAnalysis


def test_cov_type_flows():
    # given
    config = {
        "analysis": "ols_non_clustered",
        "perturbator": "constant",
        "splitter": "non_clustered",
        "cov_type": "HC1",
    }

    # when
    power_analysis = NormalPowerAnalysis.from_dict(config)

    # then
    assert power_analysis.analysis.cov_type == "HC1"


def test_cov_type_default():
    # given
    config = {
        "analysis": "ols_non_clustered",
        "perturbator": "constant",
        "splitter": "non_clustered",
    }

    # when
    power_analysis = NormalPowerAnalysis.from_dict(config)

    # then
    assert power_analysis.analysis.cov_type == "HC3"
