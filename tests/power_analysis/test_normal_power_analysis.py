import pytest

from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis, OLSAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import NormalPowerAnalysis, PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter, NonClusteredSplitter


def test_aa_power_analysis(df, analysis_gee_vainilla):
    # given
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=analysis_gee_vainilla,
        n_simulations=3,
        seed=20240922,
    )
    # when
    power = pw.power_analysis(df)
    # then
    assert abs(power - 0.05) < 0.01


def test_normal_power_sorted(df, analysis_mlm):
    # given
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=analysis_mlm,
        n_simulations=1,
        seed=20240922,
    )

    # when
    power = pw.power_line(df, average_effects=[0.05, 0.1, 0.2])
    # then
    assert power[0.05] < power[0.1]
    assert power[0.1] < power[0.2]


def test_left_power_analysis(df):
    # given
    sw = NonClusteredSplitter()

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=OLSAnalysis(),
        n_simulations=3,
        seed=20240922,
    )

    pw_left = NormalPowerAnalysis(
        splitter=sw,
        analysis=OLSAnalysis(
            hypothesis="greater",
        ),
        n_simulations=3,
        seed=20240922,
    )

    # when
    power = pw.power_line(df, average_effects=[0.05, 0.1, 0.2])
    power_left = pw_left.power_line(df, average_effects=[0.05, 0.1, 0.2])

    # then
    assert power[0.05] < power_left[0.05]
    assert power[0.1] < power_left[0.1]
    assert power[0.2] < power_left[0.2]


def test_right_power_analysis(df):
    # given
    sw = NonClusteredSplitter()

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=OLSAnalysis(),
        n_simulations=3,
        seed=20240922,
    )

    pw_right = NormalPowerAnalysis(
        splitter=sw,
        analysis=OLSAnalysis(
            hypothesis="less",
        ),
        n_simulations=3,
        seed=20240922,
    )

    # when
    power = pw.power_line(df, average_effects=[0.05, 0.1, 0.2])
    power_right = pw_right.power_line(df, average_effects=[0.05, 0.1, 0.2])

    # then
    assert power[0.05] > power_right[0.05]
    assert power[0.1] > power_right[0.1]
    assert power[0.2] > power_right[0.2]


@pytest.mark.parametrize(
    "ols, splitter, effect",
    [
        (OLSAnalysis(), NonClusteredSplitter(), 0.1),
        (OLSAnalysis(), NonClusteredSplitter(), 0.2),
        (OLSAnalysis(hypothesis="greater"), NonClusteredSplitter(), 0.1),
        (OLSAnalysis(hypothesis="less"), NonClusteredSplitter(), 0.1),
        (
            ClusteredOLSAnalysis(
                cluster_cols=["cluster", "date"],
            ),
            ClusteredSplitter(cluster_cols=["cluster", "date"]),
            0.1,
        ),
        (
            ClusteredOLSAnalysis(
                cluster_cols=["cluster", "date"],
            ),
            ClusteredSplitter(cluster_cols=["cluster", "date"]),
            0.2,
        ),
    ],
)
def test_power_sim_compare(df, ols, splitter, effect):
    # given
    perturbator = ConstantPerturbator()

    pw = PowerAnalysis(
        splitter=splitter,
        analysis=ols,
        perturbator=perturbator,
        n_simulations=200,
        seed=20240922,
    )

    pw_normal = NormalPowerAnalysis(
        splitter=splitter,
        analysis=ols,
        n_simulations=5,
        seed=20240922,
    )

    # when
    power = pw.power_line(df, average_effects=[effect])
    power_normal = pw_normal.power_line(df, average_effects=[effect])

    # then
    assert abs(power[effect] - power_normal[effect]) < 0.05


@pytest.mark.parametrize(
    "ols, splitter, effect",
    [
        (
            ClusteredOLSAnalysis(
                cluster_cols=["cluster", "date"],
            ),
            ClusteredSplitter(cluster_cols=["cluster", "date"]),
            0.2,
        ),
        (
            ClusteredOLSAnalysis(
                cluster_cols=["cluster", "date"],
            ),
            ClusteredSplitter(cluster_cols=["cluster", "date"]),
            0.5,
        ),
        (
            # using a covariate
            ClusteredOLSAnalysis(
                cluster_cols=["cluster", "date"], covariates=["cluster_id"]
            ),
            ClusteredSplitter(cluster_cols=["cluster", "date"]),
            0.5,
        ),
    ],
)
def test_power_sim_compare_cluster(correlated_df, ols, splitter, effect):
    # given
    perturbator = ConstantPerturbator()

    pw = PowerAnalysis(
        splitter=splitter,
        analysis=ols,
        perturbator=perturbator,
        n_simulations=200,
        seed=20240922,
    )

    pw_normal = NormalPowerAnalysis(
        splitter=splitter,
        analysis=ols,
        n_simulations=5,
        seed=20240922,
    )

    # when
    power = pw.power_line(correlated_df, average_effects=[effect])
    power_normal = pw_normal.power_line(correlated_df, average_effects=[effect])

    # then
    assert abs(power[effect] - power_normal[effect]) < 0.05


def test_from_config(df):
    # given
    pw_normal = NormalPowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "n_simulations": 5,
            "seed": 20240922,
        }
    )

    pw_normal_default = NormalPowerAnalysis(
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(),
        n_simulations=5,
        seed=20240922,
    )

    # when
    power_normal = pw_normal.power_line(df, average_effects=[0.1])
    power_normal_default = pw_normal_default.power_line(df, average_effects=[0.1])

    # then
    assert abs(power_normal[0.1] - power_normal_default[0.1]) < 0.03


def test_get_standard_error_hypothesis_wrong_input():
    # Check if the ValueError is raised when the hypothesis is not valid
    with pytest.raises(ValueError) as excinfo:
        NormalPowerAnalysis(
            splitter=NonClusteredSplitter(),
            analysis=OLSAnalysis(
                hypothesis="greaters",
            ),
            n_simulations=3,
            seed=20240922,
        )._normal_power_calculation(
            alpha=0.05,
            std_error=0.1,
            average_effect=0.1,
        )
    # Check if the error message is as expected
    assert "'greaters' is not a valid HypothesisEntries" in str(excinfo.value)


def test_get_mde_hypothesis_wrong_input(df):
    # Check if the ValueError is raised when the hypothesis is not valid
    with pytest.raises(ValueError) as excinfo:
        NormalPowerAnalysis(
            splitter=NonClusteredSplitter(),
            analysis=OLSAnalysis(
                hypothesis="greaters",
            ),
            n_simulations=3,
            seed=20240922,
        ).mde(
            df,
            alpha=0.05,
            power=0.7,
        )
    # Check if the error message is as expected
    assert "'greaters' is not a valid HypothesisEntries" in str(excinfo.value)


@pytest.mark.parametrize(
    "hypothesis",
    [
        "greater",
        "less",
        "two-sided",
    ],
)
def test_mde_power(df, hypothesis):
    # given
    pw_normal = NormalPowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "n_simulations": 5,
            "hypothesis": hypothesis,
            "seed": 20240922,
        }
    )

    # when
    mde = pw_normal.mde(df, power=0.9)

    power = pw_normal.power_analysis(df, average_effect=mde)

    # then
    assert abs(power - 0.9) < 0.03


@pytest.mark.parametrize(
    "hypothesis",
    [
        "greater",
        "less",
        "two-sided",
    ],
)
def test_power_mde(df, hypothesis):
    # given
    pw_normal = NormalPowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "n_simulations": 5,
            "hypothesis": hypothesis,
            "seed": 20240922,
        }
    )

    # when
    power = pw_normal.power_analysis(df, average_effect=0.1)

    mde = pw_normal.mde(df, power=power)

    # then
    assert abs(mde - 0.1) < 0.03


def test_mde_power_line(df):
    # given
    pw_normal = NormalPowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "n_simulations": 5,
            "hypothesis": "two-sided",
            "seed": 20240922,
        }
    )

    # when
    mde_power_line = pw_normal.mde_power_line(df, powers=[0.9, 0.8, 0.7])

    # then
    assert mde_power_line[0.9] > mde_power_line[0.8]
    assert mde_power_line[0.8] > mde_power_line[0.7]
