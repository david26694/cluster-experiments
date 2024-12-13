import pandas as pd

from ab_lab.experiment_analysis import SyntheticControlAnalysis
from ab_lab.perturbator import ConstantPerturbator
from ab_lab.power_analysis import PowerAnalysisWithPreExperimentData
from ab_lab.random_splitter import FixedSizeClusteredSplitter
from ab_lab.synthetic_control_utils import generate_synthetic_control_data


def test_power_analysis_with_pre_experiment_data():
    df = generate_synthetic_control_data(10, "2022-01-01", "2022-01-30")

    sw = FixedSizeClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=0.3,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    )

    pw = PowerAnalysisWithPreExperimentData(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    power = pw.power_analysis(df)
    pw.power_line(df, average_effects=[0.3, 0.4])
    assert 0 <= power <= 1
    values = list(pw.power_line(df, average_effects=[0.3, 0.4]).values())
    assert all(0 <= value <= 1 for value in values)


def test_simulate_point_estimate():
    df = generate_synthetic_control_data(10, "2022-01-01", "2022-01-30")

    sw = FixedSizeClusteredSplitter(n_treatment_clusters=1, cluster_cols=["user"])

    perturbator = ConstantPerturbator(
        average_effect=10,
    )

    analysis = SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    )

    pw = PowerAnalysisWithPreExperimentData(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    point_estimates = list(pw.simulate_point_estimate(df))
    assert (
        8 <= pd.Series(point_estimates).mean() <= 11
    ), "Point estimate is too far from the real effect."
