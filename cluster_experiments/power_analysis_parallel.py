import logging
from typing import Dict, Generator, Iterable, List, Optional, Callable
from functools import partial

from pathos.multiprocessing import Pool

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from cluster_experiments.cupac import CupacHandler
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
from cluster_experiments.utils import _get_mapping_key


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
        control: value of treatment_col considered to be control (not treatment)
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
        control: str = "A",
        n_simulations: int = 100,
        alpha: float = 0.05,
        features_cupac_model: Optional[List[str]] = None,
    ):
        self.perturbator = perturbator
        self.splitter = splitter
        self.analysis = analysis
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.treatment = treatment
        self.control = control
        self.treatment_col = treatment_col
        self.alpha = alpha

        self.cupac_handler = CupacHandler(
            cupac_model=cupac_model,
            target_col=target_col,
            features_cupac_model=features_cupac_model,
        )

        self.check_inputs()
    
    # PARALLEL EXECUTION

    def _get_single_prepared_df(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        
        df = df.copy()
        df = self.cupac_handler.add_covariates(df, pre_experiment_df)
        treatment_df = self.splitter.assign_treatment_df(df)
        self.log_nulls(treatment_df)
        # The second query allows as to do power analysis for multivariate testing
        # It assumes that we give, to each treatment value, the same number of samples
        # If this is not the case, several PowerAnalysis should be run with different weights
        treatment_df = treatment_df.query(
            f"{self.treatment_col}.notnull()", engine="python"
        ).query(
            f"{self.treatment_col}.isin(['{self.treatment}', '{self.control}'])",
            engine="python",
        )
        perturbed_df = self.perturbator.perturbate(
            treatment_df, average_effect=average_effect
        )
        return perturbed_df
    
    def _get_single_p_value(
        self,
        prepared_df: pd.DataFrame,
    ) -> float:
        return self.analysis.get_pvalue(prepared_df)
    
    def _get_single_pvalue_simulation(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        average_effect: Optional[float] = None
    ) -> float:
        prepared_df = self._get_single_prepared_df(
            df=df,
            pre_experiment_df=pre_experiment_df,
            average_effect=average_effect
        )
        p_value = self._get_single_p_value(
            prepared_df=prepared_df
        )
        return p_value

    def _get_iterable_simulator(
        self,
    
        pre_experiment_df:pd.DataFrame,
        average_effect:float
    ) -> Callable:
        iterable_function = partial(
            self._get_single_pvalue_simulation,
            pre_experiment_df=pre_experiment_df,
            average_effect=average_effect
        )
        return iterable_function

    def _get_data_generator(
        self,
        df:pd.DataFrame,
        n_simulations:int=100
    ) -> Generator[pd.DataFrame, None, None]:
        for _ in range(n_simulations): yield df.copy()

    def run_simulation(
        self,
        iterable_function:Callable,
        data_generator:Generator,
        n_processes:int=5
    ) -> np.ndarray:
        with Pool(n_processes) as pool:
            simulation_results = pool.map(iterable_function, data_generator)
        return np.array(simulation_results)

    def run_pvalue_simulation(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        average_effect: Optional[float] = None,
        n_simulations: Optional[int] = 100,
        n_processes: Optional[int] = 5
    ) -> np.ndarray:
        iterable_simulator = self._get_iterable_simulator(
            pre_experiment_df=pre_experiment_df,
            average_effect=average_effect
        )
        data_generator = self._get_data_generator(
            df=df,
            n_simulations=n_simulations
        )
        simulation_results = self.run_simulation(
            iterable_function=iterable_simulator,
            data_generator=data_generator,
            n_processes=n_processes
        )
        return simulation_results


    def log_nulls(self, df: pd.DataFrame) -> None:
        """Warns about dropping nulls in treatment column"""
        n_nulls = len(df.query(f"{self.treatment_col}.isnull()", engine="python"))
        if n_nulls > 0:
            logging.warning(
                f"There are {n_nulls} null values in treatment, dropping them"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PowerAnalysis":
        """Constructs PowerAnalysis from dictionary"""
        config = PowerConfig(**config_dict)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: PowerConfig) -> "PowerAnalysis":
        """Constructs PowerAnalysis from PowerConfig"""
        perturbator_cls = _get_mapping_key(perturbator_mapping, config.perturbator)
        splitter_cls = _get_mapping_key(splitter_mapping, config.splitter)
        analysis_cls = _get_mapping_key(analysis_mapping, config.analysis)
        cupac_cls = _get_mapping_key(cupac_model_mapping, config.cupac_model)
        return cls(
            perturbator=perturbator_cls.from_config(config),
            splitter=splitter_cls.from_config(config),
            analysis=analysis_cls.from_config(config),
            cupac_model=cupac_cls.from_config(config),
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
        )

    def check_treatment_col(self):
        """Checks consistency of treatment column"""
        assert (
            self.analysis.treatment_col == self.perturbator.treatment_col
        ), f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in perturbator ({self.perturbator.treatment_col})"

        assert (
            self.analysis.treatment_col == self.treatment_col
        ), f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in PowerAnalysis ({self.treatment_col})"

        assert (
            self.analysis.treatment_col == self.splitter.treatment_col
        ), f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in splitter ({self.splitter.treatment_col})"

    def check_target_col(self):
        assert (
            self.analysis.target_col == self.perturbator.target_col
        ), f"target_col in analysis ({self.analysis.target_col}) must be the same as target_col in perturbator ({self.perturbator.target_col})"

        assert (
            self.analysis.target_col == self.target_col
        ), f"target_col in analysis ({self.analysis.target_col}) must be the same as target_col in PowerAnalysis ({self.target_col})"

    def check_treatment(self):
        assert (
            self.analysis.treatment == self.perturbator.treatment
        ), f"treatment in analysis ({self.analysis.treatment}) must be the same as treatment in perturbator ({self.perturbator.treatment})"

        assert (
            self.analysis.treatment == self.treatment
        ), f"treatment in analysis ({self.analysis.treatment}) must be the same as treatment in PowerAnalysis ({self.treatment})"

        assert (
            self.analysis.treatment in self.splitter.treatments
        ), f"treatment in analysis ({self.analysis.treatment}) must be in treatments in splitter ({self.splitter.treatments})"

        assert (
            self.control in self.splitter.treatments
        ), f"control in power analysis ({self.control}) must be in treatments in splitter ({self.splitter.treatments})"

    def check_covariates(self):
        if hasattr(self.analysis, "covariates"):
            cupac_in_covariates = (
                self.cupac_handler.cupac_outcome_name in self.analysis.covariates
            )

            assert cupac_in_covariates or not self.cupac_handler.is_cupac, (
                f"covariates in analysis must contain {self.cupac_handler.cupac_outcome_name} if cupac_model is not None. "
                f"If you want to use cupac_model, you must add the cupac outcome to the covariates of the analysis "
                f"You may want to do covariates=['{self.cupac_handler.cupac_outcome_name}'] in your analysis method or your config"
            )

            if hasattr(self.splitter, "cluster_cols"):
                if set(self.analysis.covariates).intersection(
                    set(self.splitter.cluster_cols)
                ):
                    logging.warning(
                        f"covariates in analysis ({self.analysis.covariates}) are also cluster_cols in splitter ({self.splitter.cluster_cols}). "
                        f"Be specially careful when using switchback splitters, since the time splitter column is being overriden"
                    )

    def check_clusters(self):
        has_analysis_clusters = hasattr(self.analysis, "cluster_cols")
        has_splitter_clusters = hasattr(self.splitter, "cluster_cols")
        not_cluster_cols_cond = not has_analysis_clusters or not has_splitter_clusters
        assert (
            not_cluster_cols_cond
            or self.analysis.cluster_cols == self.splitter.cluster_cols
        ), f"cluster_cols in analysis ({self.analysis.cluster_cols}) must be the same as cluster_cols in splitter ({self.splitter.cluster_cols})"

        assert (
            has_splitter_clusters
            or not has_analysis_clusters
            or not self.analysis.cluster_cols
        ), "analysis has cluster_cols but splitter does not."

        assert (
            has_analysis_clusters
            or not has_splitter_clusters
            or not self.splitter.cluster_cols
        ), "splitter has cluster_cols but analysis does not."

        has_time_col = hasattr(self.splitter, "time_col")
        assert not (
            has_time_col
            and has_splitter_clusters
            and self.splitter.time_col not in self.splitter.cluster_cols
        ), "in switchback splitters, time_col must be in cluster_cols"

    def check_inputs(self):
        self.check_covariates()
        self.check_treatment_col()
        self.check_target_col()
        self.check_treatment()
        self.check_clusters()