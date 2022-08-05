import pandas as pd

from cluster_experiments.perturbator import Perturbator
from cluster_experiments.random_splitter import RandomSplitter


def analysis(
    df: pd.DataFrame, target: str, treatment: str, treatment_col: str
) -> float:
    df = df.copy()
    return 0.05


class PowerAnalysis:
    def __init__(
        self,
        perturbator: Perturbator,
        splitter: RandomSplitter,
        target: str,
        treatment: str,
        treatment_col: str,
        n_simulations: int = 100,
        alpha: float = 0.05,
    ):
        self.perturbator = perturbator
        self.splitter = splitter
        self.n_simulations = n_simulations
        self.target = target
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.alpha = alpha

    def power_analysis(self, df: pd.DataFrame) -> float:
        n_detected_mde = 0
        for _ in range(self.n_simulations):
            df = self.perturbator.perturbate(df)
            p_value = analysis(df, self.target, self.treatment, self.treatment_col)
            n_detected_mde += p_value < self.alpha
        return n_detected_mde / self.n_simulations
