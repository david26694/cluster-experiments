import logging
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from tqdm import tqdm

from cluster_experiments.cupac import EmptyRegressor
from cluster_experiments.experiment_analysis import ExperimentAnalysis
from cluster_experiments.perturbator import Perturbator
from cluster_experiments.power_config import (
    PowerConfig,
    analysis_mapping,
    cupac_model_mapping,
    perturbator_mapping,
    splitter_mapping,
)
from cluster_experiments.random_splitter import RandomSplitter


class PowerAnalysis:
    """
    Class used to run Power analysis. It does so by running simulations. In each simulation:
    1. Assign treatment to dataframe randomly
    2. Perturbate dataframe
    3. Add pre-experiment data if needed
    4. Run analysis

    Finally it returns the power of the analysis by counting how many times the effect was detected.

    Args:
        perturbator: Perturbator class to perturbate dataframe with treatment assigned.
        splitter: RandomSplitter class to randomly assign treatment to dataframe.
        analysis: ExperimentAnalysis class to use for analysis.
        cupac_model: Sklearn estimator class to add pre-experiment data to dataframe. If None, no pre-experiment data will be added.
        target_col: Name of the column with the outcome variable.
        treatment_col: Name of the column with the treatment variable.
        treatment: value of treatment_col considered to be treatment (not control)
        n_simulations: Number of simulations to run.
        alpha: Significance level.
        features_cupac_model: Covariates to be used in cupac model

    Usage:
    ```python
    from datetime import date

    import numpy as np
    import pandas as pd
    from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
    from cluster_experiments.perturbator import UniformPerturbator
    from cluster_experiments.power_analysis import PowerAnalysis
    from cluster_experiments.random_splitter import ClusteredSplitter

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
    sw = ClusteredSplitter(
        cluster_cols=["cluster", "date"],
    )

    perturbator = UniformPerturbator()

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    pw = PowerAnalysis(
        perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
    )

    power = pw.power_analysis(df, average_effect=0.1)
    print(f"{power = }")
    ```
    """

    def __init__(
        self,
        perturbator: Perturbator,
        splitter: RandomSplitter,
        analysis: ExperimentAnalysis,
        cupac_model: Optional[BaseEstimator] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        n_simulations: int = 100,
        alpha: float = 0.05,
        features_cupac_model: Optional[List[str]] = None,
    ):
        self.perturbator = perturbator
        self.splitter = splitter
        self.analysis = analysis
        self.cupac_model: BaseEstimator = cupac_model or EmptyRegressor()
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.cupac_outcome_name = f"estimate_{self.target_col}"
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.alpha = alpha
        self.features_cupac_model: List[str] = features_cupac_model or []
        self.is_cupac = not isinstance(self.cupac_model, EmptyRegressor)

        self.check_inputs()

    def power_analysis(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
    ) -> float:
        """
        Run power analysis by simulation
        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
        """
        df = df.copy()

        if pre_experiment_df is not None and self.is_cupac:
            df = self.add_covariates(df, pre_experiment_df)

        n_detected_mde = 0

        for _ in tqdm(range(self.n_simulations), disable=not verbose):
            treatment_df = self.splitter.assign_treatment_df(df)
            self.log_nulls(treatment_df)
            treatment_df = treatment_df.query(
                f"{self.treatment_col}.notnull()", engine="python"
            )
            treatment_df = self.perturbator.perturbate(
                treatment_df, average_effect=average_effect
            )
            p_value = self.analysis.get_pvalue(treatment_df)
            n_detected_mde += p_value < self.alpha

        return n_detected_mde / self.n_simulations

    def _prep_data_cupac(
        self, df: pd.DataFrame, pre_experiment_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Prepares data for training and prediction"""
        df = df.copy()
        pre_experiment_df = pre_experiment_df.copy()
        df_predict = df.drop(columns=[self.target_col])
        # Split data into X and y
        pre_experiment_x = pre_experiment_df.drop(columns=[self.target_col])
        pre_experiment_y = pre_experiment_df[self.target_col]

        # Keep only cupac features
        if self.features_cupac_model:
            pre_experiment_x = pre_experiment_x[self.features_cupac_model]
            df_predict = df_predict[self.features_cupac_model]

        return df_predict, pre_experiment_x, pre_experiment_y

    def add_covariates(
        self, df: pd.DataFrame, pre_experiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Train model to predict outcome variable (based on pre-experiment data)
        and  add the prediction to the experiment dataframe
        Args:
            pre_experiment_df: Dataframe with pre-experiment data.
            df: Dataframe with outcome and treatment variables.

        """
        df = df.copy()
        pre_experiment_df = pre_experiment_df.copy()
        df_predict, pre_experiment_x, pre_experiment_y = self._prep_data_cupac(
            df=df, pre_experiment_df=pre_experiment_df
        )

        # Fit model
        self.cupac_model.fit(pre_experiment_x, pre_experiment_y)

        # Predict
        if isinstance(self.cupac_model, RegressorMixin):
            estimated_target = self.cupac_model.predict(df_predict)
        elif isinstance(self.cupac_model, ClassifierMixin):
            estimated_target = self.cupac_model.predict_proba(df_predict)[:, 1]
        else:
            raise ValueError(
                "cupac_model should be an instance of RegressorMixin or ClassifierMixin"
            )

        # Add cupac outcome name to df
        df[self.cupac_outcome_name] = estimated_target
        return df

    def log_nulls(self, df: pd.DataFrame) -> None:
        """Warns about dropping nulls in treatment column"""
        n_nulls = len(df.query(f"{self.treatment_col}.isnull()", engine="python"))
        if n_nulls > 0:
            logging.warning(
                f"There are {n_nulls} null values in treatment, dropping them"
            )

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
        cupac_model = cls._get_mapping_key(
            cupac_model_mapping, config.cupac_model
        ).from_config(config)
        return cls(
            perturbator=perturbator,
            splitter=splitter,
            analysis=analysis,
            cupac_model=cupac_model,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
        )

    def check_inputs(self):
        cupac_not_in_covariates = (
            self.cupac_outcome_name not in self.analysis.covariates
        )

        if self.is_cupac and cupac_not_in_covariates:
            raise ValueError(
                f"covariates in analysis must contain {self.cupac_outcome_name} if cupac_model is not None. "
                f"If you want to use cupac_model, you must add the cupac outcome to the covariates of the analysis "
                f"You may want to do covariates=['{self.cupac_outcome_name}'] in your analysis method or your config"
            )

        if self.analysis.target_col != self.perturbator.target_col:
            raise ValueError(
                f"target_col in analysis ({self.analysis.target_col}) must be the same as target_col in perturbator ({self.perturbator.target_col})"
            )

        if self.analysis.target_col != self.target_col:
            raise ValueError(
                f"target_col in analysis ({self.analysis.target_col}) must be the same as target_col in PowerAnalysis ({self.target_col})"
            )

        if self.analysis.treatment_col != self.perturbator.treatment_col:
            raise ValueError(
                f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in perturbator ({self.perturbator.treatment_col})"
            )

        if self.analysis.treatment_col != self.treatment_col:
            raise ValueError(
                f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in PowerAnalysis ({self.treatment_col})"
            )

        if self.analysis.treatment != self.perturbator.treatment:
            raise ValueError(
                f"treatment in analysis ({self.analysis.treatment}) must be the same as treatment in perturbator ({self.perturbator.treatment})"
            )

        if self.analysis.treatment_col != self.splitter.treatment_col:
            raise ValueError(
                f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in splitter ({self.splitter.treatment_col})"
            )

        has_analysis_clusters = hasattr(self.analysis, "cluster_cols")
        has_splitter_clusters = hasattr(self.splitter, "cluster_cols")
        cluster_cols_cond = has_analysis_clusters and has_splitter_clusters
        if (
            cluster_cols_cond
            and self.analysis.cluster_cols != self.splitter.cluster_cols
        ):
            raise ValueError(
                f"cluster_cols in analysis ({self.analysis.cluster_cols}) must be the same as cluster_cols in splitter ({self.splitter.cluster_cols})"
            )

        if (
            has_analysis_clusters
            and not has_splitter_clusters
            and self.analysis.cluster_cols
        ):
            raise ValueError("analysis has cluster_cols but splitter does not.")

        if (
            not has_analysis_clusters
            and has_splitter_clusters
            and self.splitter.cluster_cols
        ):
            raise ValueError("splitter has cluster_cols but analysis does not.")
