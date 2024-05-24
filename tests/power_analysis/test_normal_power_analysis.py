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


def test_power_sim_compare_cluster(df):
    from datetime import date

    import numpy as np
    import pandas as pd

    N = 10_000
    clusters = [f"Cluster {i}" for i in range(10)]
    dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 15)]
    df = pd.DataFrame(
        {
            "cluster": np.random.choice(clusters, size=N),
            "date": np.random.choice(dates, size=N),
        }
    ).assign(
        # Target is a linear combination of cluster and day of week, plus some noise
        cluster_id=lambda df: df["cluster"].astype("category").cat.codes,
        day_of_week=lambda df: pd.to_datetime(df["date"]).dt.dayofweek,
        target=lambda df: df["cluster_id"]
        + df["day_of_week"]
        + np.random.normal(size=N),
    )

    splitter = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    ols = ClusteredOLSAnalysis(
        cluster_cols=["cluster", "date"],
    )
    effect = 0.2

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
