import pytest

from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis, OLSAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import NormalPowerAnalysis, PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter, NonClusteredSplitter


def test_aa_power_analysis(df, analysis_gee_vainilla):
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=analysis_gee_vainilla,
        n_simulations=3,
        seed=20240922,
    )

    power = pw.power_analysis(df)
    assert abs(power - 0.05) < 0.01


def test_normal_power_sorted(df, analysis_mlm):
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    pw = NormalPowerAnalysis(
        splitter=sw,
        analysis=analysis_mlm,
        n_simulations=3,
        seed=20240922,
    )

    power = pw.power_line(df, average_effects=[0.05, 0.1, 0.2])
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
