from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysisWithPreExperimentData
from cluster_experiments.random_splitter import FixedSizeClusteredSplitter
from cluster_experiments.synthetic_control_utils import generate_data

df = generate_data(10, "2022-01-01", "2022-01-30")

sw = FixedSizeClusteredSplitter(n_treatment_clusters=2, cluster_cols=["user"])

perturbator = ConstantPerturbator(
    average_effect=0.1,
)

analysis = SyntheticControlAnalysis(
    cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
)

pw = PowerAnalysisWithPreExperimentData(
    perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
)

power = pw.power_analysis(df)
print(f"{power = }")
print(pw.power_line(df, average_effects=[0.1, 0.2, 0.5, 1, 1.5], n_jobs=-1))
print(pw.simulate_point_estimate(df))
