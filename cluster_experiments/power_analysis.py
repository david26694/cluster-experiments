from typing import Optional

import pandas as pd

from cluster_experiments.experiment_analysis import ExperimentAnalysis
from cluster_experiments.perturbator import Perturbator
from cluster_experiments.power_config import (
    PowerConfig,
    analysis_mapping,
    featurizer_mapping,
    perturbator_mapping,
    splitter_mapping,
)
from cluster_experiments.pre_experiment_covariates import PreExperimentFeaturizer
from cluster_experiments.random_splitter import RandomSplitter


class PowerAnalysis:
    """Class used to run Power analysis. It does so by running simulations. In each simulation:
    1. Assign treatment to dataframe randomly
    2. Perturbate dataframe
    3. Add pre-experiment data if needed
    4. Run analysis

    Finally it returns the power of the analysis by counting how many times the effect was detected.
    """

    def __init__(
        self,
        perturbator: Perturbator,
        splitter: RandomSplitter,
        analysis: ExperimentAnalysis,
        featurizer: Optional[PreExperimentFeaturizer] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        n_simulations: int = 100,
        alpha: float = 0.05,
    ):
        """
        Initilize PowerAnalysis class.
        Args:
            perturbator: Perturbator class to perturbate dataframe with treatment assigned.
            splitter: RandomSplitter class to randomly assign treatment to dataframe.
            analysis: ExperimentAnalysis class to use for analysis.
            featurizer: PreExperimentFeaturizer class to add pre-experiment data to dataframe. If None, no pre-experiment data will be added.
            target_col: Name of the column with the outcome variable.
            treatment_col: Name of the column with the treatment variable.
            treatment: value of treatment_col considered to be treatment (not control)
            n_simulations: Number of simulations to run.
            alpha: Significance level.

        Usage:
        ```python
        from datetime import date

        import numpy as np
        import pandas as pd
        from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
        from cluster_experiments.perturbator import UniformPerturbator
        from cluster_experiments.power_analysis import PowerAnalysis
        from cluster_experiments.random_splitter import SwitchbackSplitter

        N = 1_000
        users = [f"User {i}" for i in range(1000)]
        clusters = [f"Cluster {i}" for i in range(100)]
        dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
        df = pd.DataFrame(
            {
                "cluster": np.random.choice(clusters, size=N),
                "target": np.random.normal(0, 1, size=N),
                "user": np.random.choice(users, size=N),
                "date": np.random.choice(dates, size=N),
            }
        )

        experiment_dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(15, 32)]
        sw = SwitchbackSplitter(
            clusters=clusters,
            dates=experiment_dates,
        )

        perturbator = UniformPerturbator(
            average_effect=0.1,
        )

        analysis = GeeExperimentAnalysis(
            cluster_cols=["cluster", "date"],
        )

        pw = PowerAnalysis(
            perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
        )

        power = pw.power_analysis(df)
        print(f"{power = }")
        ```

        """
        self.perturbator = perturbator
        self.splitter = splitter
        self.analysis = analysis
        self.featurizer: PreExperimentFeaturizer = (
            featurizer or PreExperimentFeaturizer()
        )
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.alpha = alpha

    def power_analysis(
        self, df: pd.DataFrame, pre_experiment_df: Optional[pd.DataFrame] = None
    ) -> float:
        """
        Run power analysis by simulation
        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
        """
        df = df.copy()
        n_detected_mde = 0
        if pre_experiment_df is not None and not self.featurizer.is_empty:
            self.featurizer.fit_pre_experiment_data(pre_experiment_df)
        for _ in range(self.n_simulations):
            treatment_df = self.splitter.assign_treatment_df(df)
            treatment_df = treatment_df.query(f"{self.treatment_col}.notnull()")
            treatment_df = self.perturbator.perturbate(treatment_df)
            if not self.featurizer.is_empty:
                treatment_df = self.featurizer.add_pre_experiment_data(treatment_df)

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
        """Constructs PowerAnalysis from dictionary"""
        config = PowerConfig(**config_dict)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: PowerConfig):
        """Constructs PowerAnalysis from PowerConfig"""
        perturbator = cls._get_mapping_key(
            perturbator_mapping, config.perturbator
        ).from_config(config)
        splitter = cls._get_mapping_key(splitter_mapping, config.splitter).from_config(
            config
        )
        analysis = cls._get_mapping_key(analysis_mapping, config.analysis).from_config(
            config
        )
        featurizer = cls._get_mapping_key(
            featurizer_mapping, config.featurizer
        ).from_config(config)
        return cls(
            perturbator=perturbator,
            splitter=splitter,
            analysis=analysis,
            featurizer=featurizer,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
        )
