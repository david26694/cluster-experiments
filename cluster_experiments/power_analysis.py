from typing import Optional

import pandas as pd

from cluster_experiments.experiment_analysis import ExperimentAnalysis
from cluster_experiments.perturbator import Perturbator
from cluster_experiments.power_config import (
    PowerConfig,
    aggregator_mapping,
    analysis_mapping,
    perturbator_mapping,
    splitter_mapping,
)
from cluster_experiments.pre_experiment_covariates import Aggregator
from cluster_experiments.random_splitter import RandomSplitter


class PowerAnalysis:
    def __init__(
        self,
        perturbator: Perturbator,
        splitter: RandomSplitter,
        analysis: ExperimentAnalysis,
        aggregator: Optional[Aggregator] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        n_simulations: int = 100,
        alpha: float = 0.05,
    ):
        self.perturbator = perturbator
        self.splitter = splitter
        self.analysis = analysis
        self.aggregator: Aggregator = aggregator or Aggregator()
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.alpha = alpha

    def power_analysis(
        self, df: pd.DataFrame, pre_experiment_df: Optional[pd.DataFrame] = None
    ) -> float:
        df = df.copy()
        n_detected_mde = 0
        if pre_experiment_df is not None and not self.aggregator.is_empty:
            self.aggregator.set_pre_experiment_agg(pre_experiment_df)
        for _ in range(self.n_simulations):
            treatment_df = self.splitter.assign_treatment_df(df)
            treatment_df = treatment_df.query(f"{self.treatment_col}.notnull()")
            treatment_df = self.perturbator.perturbate(treatment_df)
            if not self.aggregator.is_empty:
                treatment_df = self.aggregator.add_pre_experiment_agg(treatment_df)

            p_value = self.analysis.get_pvalue(treatment_df)
            n_detected_mde += p_value < self.alpha
        return n_detected_mde / self.n_simulations

    @staticmethod
    def _get_mapping_key(mapping, key):
        try:
            return mapping[key]
        except KeyError:
            raise KeyError(
                f"Could not find {key = } in mapping. All options are the following: {list(mapping.keys())}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = PowerConfig(**config_dict)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: PowerConfig):
        perturbator = cls._get_mapping_key(
            perturbator_mapping, config.perturbator
        ).from_config(config)
        splitter = cls._get_mapping_key(splitter_mapping, config.splitter).from_config(
            config
        )
        analysis = cls._get_mapping_key(analysis_mapping, config.analysis).from_config(
            config
        )
        aggregator = cls._get_mapping_key(
            aggregator_mapping, config.aggregator
        ).from_config(config)
        return cls(
            perturbator=perturbator,
            splitter=splitter,
            analysis=analysis,
            aggregator=aggregator,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
        )
