import pandas as pd

from cluster_experiments.experiment_analysis import ExperimentAnalysis
from cluster_experiments.perturbator import Perturbator
from cluster_experiments.power_config import (
    PowerConfig,
    analysis_mapping,
    perturbator_mapping,
    splitter_mapping,
)
from cluster_experiments.random_splitter import RandomSplitter


class PowerAnalysis:
    def __init__(
        self,
        perturbator: Perturbator,
        splitter: RandomSplitter,
        analysis: ExperimentAnalysis,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        n_simulations: int = 100,
        alpha: float = 0.05,
    ):
        self.perturbator = perturbator
        self.splitter = splitter
        self.analysis = analysis
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.alpha = alpha

    def power_analysis(self, df: pd.DataFrame) -> float:
        df = df.copy()
        n_detected_mde = 0
        for _ in range(self.n_simulations):
            treatment_df = self.splitter.assign_treatment_df(df)
            treatment_df = treatment_df.query(f"{self.treatment_col}.notnull()")
            treatment_df = self.perturbator.perturbate(treatment_df)
            p_value = self.analysis.get_pvalue(treatment_df)
            n_detected_mde += p_value < self.alpha
        return n_detected_mde / self.n_simulations

    @classmethod
    def from_config(cls, config: PowerConfig):
        perturbator = perturbator_mapping[config.perturbator].from_config(config)
        splitter = splitter_mapping[config.splitter].from_config(config)
        analysis = analysis_mapping[config.analysis].from_config(config)
        return cls(
            perturbator=perturbator,
            splitter=splitter,
            analysis=analysis,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
        )
