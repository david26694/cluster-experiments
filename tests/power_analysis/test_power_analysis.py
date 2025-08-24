from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.power_config import PowerConfig
from cluster_experiments.random_splitter import ClusteredSplitter


def test_power_analysis(df, perturbator, analysis_gee_vainilla):
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    pw = PowerAnalysis(
        perturbator=perturbator,
        splitter=sw,
        analysis=analysis_gee_vainilla,
        n_simulations=3,
    )

    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1


def test_power_analysis_config(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
        average_effect=0.0,
    )
    pw = PowerAnalysis.from_config(config)
    power = pw.power_analysis(df)
    assert power >= 0
    assert power <= 1


def test_power_analysis_config_avg_effect(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_config(config)
    power = pw.power_analysis(df, average_effect=0.0)
    assert power >= 0
    assert power <= 1


def test_power_analysis_dict(df):
    config = dict(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)
    power = pw.power_analysis(df, average_effect=0.0)
    assert power >= 0
    assert power <= 1

    power_verbose = pw.power_analysis(df, verbose=True, average_effect=0.0)
    assert power_verbose >= 0
    assert power_verbose <= 1


def test_different_names(df):
    df = df.rename(
        columns={
            "cluster": "cluster_0",
            "target": "target_0",
            "date": "date_0",
        }
    )
    config = dict(
        cluster_cols=["cluster_0", "date_0"],
        analysis="ols_clustered",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
        treatment_col="treatment_0",
        target_col="target_0",
    )
    pw = PowerAnalysis.from_dict(config)
    power = pw.power_analysis(df, average_effect=0.0)
    assert power >= 0
    assert power <= 1

    power_verbose = pw.power_analysis(df, verbose=True, average_effect=0.0)
    assert power_verbose >= 0
    assert power_verbose <= 1


def test_ttest(df):
    config = dict(
        cluster_cols=["cluster", "date"],
        analysis="ttest_clustered",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)
    power = pw.power_analysis(df, average_effect=0.0)
    assert power >= 0
    assert power <= 1

    power_verbose = pw.power_analysis(df, verbose=True, average_effect=0.0)
    assert power_verbose >= 0
    assert power_verbose <= 1


def test_paired_ttest(df):
    config = dict(
        cluster_cols=["cluster", "date"],
        strata_cols=["cluster"],
        analysis="paired_ttest_clustered",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)

    power = pw.power_analysis(df, average_effect=0.0)
    assert power >= 0
    assert power <= 1

    power_verbose = pw.power_analysis(df, verbose=True, average_effect=0.0)
    assert power_verbose >= 0
    assert power_verbose <= 1


def test_power_alpha(df):
    config = PowerConfig(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=10,
        average_effect=0.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    power_50 = pw.power_analysis(df, alpha=0.5, verbose=True)
    power_01 = pw.power_analysis(df, alpha=0.01)

    assert power_50 > power_01


def test_length_simulation(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="ols_clustered",
        perturbator="constant",
        splitter="clustered",
        n_simulations=10,
        average_effect=0.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    i = 0
    for _ in pw.simulate_pvalue(df, n_simulations=5):
        i += 1
    assert i == 5


def test_point_estimate_gee(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        n_simulations=10,
        average_effect=5.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    for point_estimate in pw.simulate_point_estimate(df, n_simulations=1):
        assert point_estimate > 0.0


def test_point_estimate_clustered_ols(df):
    config = PowerConfig(
        cluster_cols=["cluster", "date"],
        analysis="ols_clustered",
        perturbator="constant",
        splitter="clustered",
        n_simulations=10,
        average_effect=5.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    for point_estimate in pw.simulate_point_estimate(df, n_simulations=1):
        assert point_estimate > 0.0


def test_point_estimate_ols(df):
    config = PowerConfig(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=10,
        average_effect=5.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    for point_estimate in pw.simulate_point_estimate(df, n_simulations=1):
        assert point_estimate > 0.0


def test_power_line(df):
    config = PowerConfig(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=10,
        average_effect=0.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)

    power_line = pw.power_line(df, average_effects=[0.0, 1.0], n_simulations=10)
    assert len(power_line) == 2
    assert power_line[0.0] >= 0
    assert power_line[1.0] >= power_line[0.0]


def test_running_power(df):
    # given
    config = PowerConfig(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        n_simulations=50,
        average_effect=0.0,
        alpha=0.05,
    )
    pw = PowerAnalysis.from_config(config)
    previous_power = 0
    for i, power in enumerate(pw.running_power_analysis(df, average_effect=0.1)):
        # when: we've run enough simulations
        if i > 20:
            # then: the power should be stable (no changes bigger than 0.05)
            assert abs(power - previous_power) <= 0.05
        previous_power = power


def test_hypothesis_from_dict(df):
    # given
    config_less = dict(
        analysis="ols_non_clustered",
        perturbator="constant",
        splitter="non_clustered",
        hypothesis="less",
        n_simulations=20,
    )
    pw_less = PowerAnalysis.from_dict(config_less)

    config_greater = config_less.copy()
    config_greater["hypothesis"] = "greater"
    pw_greater = PowerAnalysis.from_dict(config_greater)

    config_two_sided = config_less.copy()
    config_two_sided["hypothesis"] = "two-sided"
    pw_two_sided = PowerAnalysis.from_dict(config_two_sided)

    # when
    power_less = pw_less.power_analysis(df, average_effect=1.0)
    power_greater = pw_greater.power_analysis(df, average_effect=1.0)
    power_two_sided = pw_two_sided.power_analysis(df, average_effect=1.0)

    # then
    assert power_less < power_greater
    assert power_less < power_two_sided
