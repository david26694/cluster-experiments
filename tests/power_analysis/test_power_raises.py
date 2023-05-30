import pytest

from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter, NonClusteredSplitter


def test_raises_cupac():
    config = dict(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        cupac_model="mean_cupac_model",
        n_simulations=4,
    )
    with pytest.raises(AssertionError):
        PowerAnalysis.from_dict(config)


def test_data_checks(df):
    config = dict(
        cluster_cols=["cluster", "date"],
        analysis="gee",
        perturbator="constant",
        splitter="clustered",
        n_simulations=4,
    )
    pw = PowerAnalysis.from_dict(config)
    df["target"] = df["target"] == 1
    with pytest.raises(ValueError):
        pw.power_analysis(df, average_effect=0.0)


def test_raise_target():
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
        target_col="another_target",
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )


def test_raise_treatment():
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = ConstantPerturbator(average_effect=0.1, treatment="C")

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )


def test_raise_treatment_col():
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
        treatment_col="another_treatment",
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )


def test_raise_treatment_col_2():
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
        treatment_col="another_treatment",
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )


def test_raise_cluster_cols():
    sw = ClusteredSplitter(
        cluster_cols=["cluster"],
    )

    perturbator = ConstantPerturbator(
        average_effect=0.1,
        target_col="another_target",
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )


def test_raise_clustering_mismatch():
    sw = NonClusteredSplitter()

    perturbator = ConstantPerturbator(
        average_effect=0.1,
        target_col="another_target",
    )

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    with pytest.raises(AssertionError):
        PowerAnalysis(
            perturbator=perturbator,
            splitter=sw,
            analysis=analysis,
            n_simulations=3,
        )
