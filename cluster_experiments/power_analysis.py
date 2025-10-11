import logging
import random
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Literal, Callable, Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from tqdm import tqdm

from cluster_experiments.cupac import CupacHandler
from cluster_experiments.experiment_analysis import (
    DeltaMethodAnalysis,
    ExperimentAnalysis,
)
from cluster_experiments.perturbator import Perturbator
from cluster_experiments.power_config import (
    PowerConfig,
    analysis_mapping,
    cupac_model_mapping,
    perturbator_mapping,
    splitter_mapping,
)
from cluster_experiments.random_splitter import RandomSplitter, RepeatedSampler
from cluster_experiments.utils import HypothesisEntries, _get_mapping_key


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
        seed: Optional. Seed to use for the splitter.

    Usage:
    ```python
    from datetime import date

    import numpy as np
    import pandas as pd
    from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
    from cluster_experiments.perturbator import ConstantPerturbator
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

    perturbator = ConstantPerturbator()

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
        scale_col: Optional[str] = None,
        seed: Optional[int] = None,
        hypothesis: str = "two-sided",
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
        self.hypothesis = hypothesis
        self.scale_col = scale_col

        self.cupac_handler = CupacHandler(
            cupac_model=cupac_model,
            target_col=target_col,
            scale_col=scale_col,
            features_cupac_model=features_cupac_model,
        )
        if seed is not None:
            random.seed(seed)  # seed for splitter
            np.random.seed(seed)  # seed for the binary perturbator
            # may need to seed other stochasticity sources if added

        self.check_inputs()

    def _simulate_perturbed_df(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
        n_simulations: int = 100,
    ) -> Generator[pd.DataFrame, None, None]:
        """Yields splitted + perturbated dataframe for each iteration of the simulation."""
        df = df.copy()
        df = self.cupac_handler.add_covariates(df, pre_experiment_df)

        for _ in tqdm(range(n_simulations), disable=not verbose):
            yield self._split_and_perturbate(df, average_effect)

    def simulate_pvalue(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
        n_simulations: int = 100,
    ) -> Generator[float, None, None]:
        """
        Yields p-values for each iteration of the simulation.
        In general, this is to be used in power_analysis method. However,
        if you're interested in the distribution of p-values, you can use this method to generate them.
        Args:
            df: Dataframe with outcome variable.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
        """
        for perturbed_df in self._simulate_perturbed_df(
            df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            average_effect=average_effect,
            n_simulations=n_simulations,
        ):
            yield self.analysis.get_pvalue(perturbed_df)

    def running_power_analysis(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
        n_simulations: int = 100,
    ) -> Generator[float, None, None]:
        """
        Yields running power for each iteration of the simulation.
        if you're interested in getting the power at each iteration, you can use this method to generate them.
        Args:
            df: Dataframe with outcome variable.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
        """
        n_rejected = 0
        for i, perturbed_df in enumerate(
            self._simulate_perturbed_df(
                df,
                pre_experiment_df=pre_experiment_df,
                verbose=verbose,
                average_effect=average_effect,
                n_simulations=n_simulations,
            )
        ):
            p_value = self.analysis.get_pvalue(perturbed_df)
            n_rejected += int(p_value < self.alpha)
            yield n_rejected / (i + 1)

    def simulate_point_estimate(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
        n_simulations: int = 100,
    ) -> Generator[float, None, None]:
        """
        Yields point estimates for each iteration of the simulation.
        In general, this is to be used in power_analysis method. However,
        if you're interested in the distribution of point estimates, you can use this method to generate them.

        This is an experimental feature and it might change in the future.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
        """
        for perturbed_df in self._simulate_perturbed_df(
            df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            average_effect=average_effect,
            n_simulations=n_simulations,
        ):
            yield self.analysis.get_point_estimate(perturbed_df)

    def power_analysis(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: Optional[float] = None,
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
        n_jobs: int = 1,
    ) -> float:
        """
        Run power analysis by simulation
        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
            n_jobs: Number of jobs to run in parallel. If 1, it will run in serial.
        """
        n_simulations = self.n_simulations if n_simulations is None else n_simulations
        alpha = self.alpha if alpha is None else alpha

        df = df.copy()
        df = self.cupac_handler.add_covariates(df, pre_experiment_df)

        if n_jobs == 1:
            return self._non_parallel_loop(
                df, average_effect, n_simulations, alpha, verbose
            )
        elif n_jobs > 1 or n_jobs == -1:
            return self._parallel_loop(
                df, average_effect, n_simulations, alpha, verbose, n_jobs
            )
        else:
            raise ValueError("n_jobs must be greater than 0, or -1.")

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split dataframe.
        Args:
            df: Dataframe with outcome variable
        """
        treatment_df = self.splitter.assign_treatment_df(df)
        self.log_nulls(treatment_df)
        treatment_df = treatment_df.query(
            f"{self.treatment_col}.notnull()", engine="python"
        ).query(
            f"{self.treatment_col}.isin(['{self.treatment}', '{self.control}'])",
            engine="python",
        )

        return treatment_df

    def _perturbate(
        self, treatment_df: pd.DataFrame, average_effect: Optional[float]
    ) -> pd.DataFrame:
        """
        Perturbate dataframe using perturbator.
        Args:
            df: Dataframe with outcome variable
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
        """

        perturbed_df = self.perturbator.perturbate(
            treatment_df, average_effect=average_effect
        )
        return perturbed_df

    def _split_and_perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float]
    ) -> pd.DataFrame:
        treatment_df = self._split(df)
        perturbed_df = self._perturbate(
            treatment_df=treatment_df, average_effect=average_effect
        )
        return perturbed_df

    def _run_simulation(self, args: Tuple[pd.DataFrame, Optional[float]]) -> float:
        df, average_effect = args
        perturbed_df = self._split_and_perturbate(df, average_effect)
        return self.analysis.get_pvalue(perturbed_df)

    def _non_parallel_loop(
        self,
        df: pd.DataFrame,
        average_effect: Optional[float],
        n_simulations: int,
        alpha: float,
        verbose: bool,
    ) -> float:
        """
        Run power analysis by simulation in serial
        Args:
            df: Dataframe with outcome and treatment variables.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        n_detected_mde = 0
        for _ in tqdm(range(n_simulations), disable=not verbose):
            p_value = self._run_simulation((df, average_effect))
            if verbose:
                print(f"p_value of simulation run: {p_value:.3f}")
            n_detected_mde += p_value < alpha

        return n_detected_mde / n_simulations

    def _parallel_loop(
        self,
        df: pd.DataFrame,
        average_effect: Optional[float],
        n_simulations: int,
        alpha: float,
        verbose: bool,
        n_jobs: int,
    ) -> float:
        """
        Run power analysis by simulation in parallel
        Args:
            df: Dataframe with outcome and treatment variables.
            average_effect: Average effect of treatment. If None, it will use the perturbator average effect.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
            n_jobs: Number of jobs to run in parallel.
        """
        from multiprocessing import Pool, cpu_count

        n_jobs = n_jobs if n_jobs != -1 else cpu_count()

        n_detected_mde = 0
        with Pool(processes=n_jobs) as pool:
            args = [(df, average_effect) for _ in range(n_simulations)]
            results = pool.imap_unordered(self._run_simulation, args)
            for p_value in tqdm(results, total=n_simulations, disable=not verbose):
                n_detected_mde += p_value < alpha

        return n_detected_mde / n_simulations

    def power_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effects: Iterable[float] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
        n_jobs: int = 1,
    ) -> Dict[float, float]:
        """Runs power analysis with multiple average effects

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effects: Average effects to test.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
            n_jobs: Number of jobs to run in parallel.

        Returns:
            Dictionary with average effects as keys and power as values.
        """
        return {
            effect: self.power_analysis(
                df=df,
                pre_experiment_df=pre_experiment_df,
                verbose=verbose,
                average_effect=effect,
                n_simulations=n_simulations,
                alpha=alpha,
                n_jobs=n_jobs,
            )
            for effect in tqdm(
                list(average_effects), disable=not verbose, desc="Effects loop"
            )
        }

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
            control=config.control,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
            features_cupac_model=config.features_cupac_model,
            seed=config.seed,
            hypothesis=config.hypothesis,
            scale_col=config.scale_col,
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
            self.treatment != self.control
        ), f"treatment in PowerAnalysis ({self.treatment}) must not be the same as control in PowerAnalysis ({self.control})"

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
            or isinstance(self.splitter, RepeatedSampler)
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

    def check_scale_col(self):
        if self.scale_col is not None:
            if not isinstance(self.analysis, DeltaMethodAnalysis):
                raise ValueError(
                    "If scale_col is provided, the analysis method must be DeltaMethodAnalysis, since it is the only one that supports scale_col."
                )

    def check_inputs(self):
        self.check_covariates()
        self.check_treatment_col()
        self.check_target_col()
        self.check_treatment()
        self.check_clusters()
        self.check_scale_col()


class PowerAnalysisWithPreExperimentData(PowerAnalysis):
    """
    This is intended to work mainly for diff-in-diff or synthetic control-like estimators, and NOT for cases of CUPED/CUPAC.
    Same as PowerAnalysis, but allowing a perturbation only at experiment period and keeping pre-experiment df intact.
    Using this class, the pre experiment df is also available when the class is instantiated.
    """

    def _perturbate(
        self, treatment_df: pd.DataFrame, average_effect: Optional[float]
    ) -> pd.DataFrame:
        if not hasattr(self.analysis, "_split_pre_experiment_df"):
            raise AttributeError(
                "The PowerAnalysisWithPreExperimentData is intended to work mainly for diff-in-diff or synthetic control-like estimators."
                "For other cases use the PowerAnalysis"
            )

        df, pre_experiment_df = self.analysis._split_pre_experiment_df(treatment_df)

        perturbed_df = self.perturbator.perturbate(df, average_effect=average_effect)

        return pd.concat([perturbed_df, pre_experiment_df])


class NormalPowerAnalysis:
    """
    Class used to run Power analysis, using the central limit theorem to estimate power based on standard errors of the estimator,
    and the fact that the coefficients of a regression are normally distributed.
    It does so by running simulations. In each simulation:
    1. Assign treatment to dataframe randomly
    2. Add pre-experiment data if needed
    3. Get standard error from analysis

    Finally it returns the power of the analysis by counting how many times the effect was detected.

    Args:
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
        seed: Optional. Seed to use for the splitter.

    Usage:
    ```python
    from datetime import date

    import numpy as np
    import pandas as pd
    from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
    from cluster_experiments.power_analysis import NormalPowerAnalysis
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

    analysis = GeeExperimentAnalysis(
        cluster_cols=["cluster", "date"],
    )

    pw = NormalPowerAnalysis(
        splitter=sw, analysis=analysis, n_simulations=50
    )

    power = pw.power_analysis(df, average_effect=0.1)
    print(f"{power = }")
    ```
    """

    def __init__(
        self,
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
        scale_col: Optional[str] = None,
        seed: Optional[int] = None,
        hypothesis: str = "two-sided",
        time_col: Optional[str] = None,
    ):
        self.splitter = splitter
        self.analysis = analysis
        self.n_simulations = n_simulations
        self.target_col = target_col
        self.treatment = treatment
        self.control = control
        self.treatment_col = treatment_col
        self.alpha = alpha
        self.hypothesis = hypothesis
        self.time_col = time_col
        self.scale_col = scale_col

        self.cupac_handler = CupacHandler(
            cupac_model=cupac_model,
            target_col=target_col,
            features_cupac_model=features_cupac_model,
            scale_col=scale_col,
        )
        if seed is not None:
            random.seed(seed)  # seed for splitter
            np.random.seed(seed)  # numpy seed
            # may need to seed other stochasticity sources if added

        self.check_inputs()

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Split dataframe.
        Args:
            df: Dataframe with outcome variable
        """
        treatment_df = self.splitter.assign_treatment_df(df)
        self.log_nulls(treatment_df)
        treatment_df = treatment_df.query(
            f"{self.treatment_col}.notnull()", engine="python"
        ).query(
            f"{self.treatment_col}.isin(['{self.treatment}', '{self.control}'])",
            engine="python",
        )
        return treatment_df

    def _get_standard_error(
        self,
        df: pd.DataFrame,
        n_simulations: int,
        verbose: bool,
    ) -> Generator[float, None, None]:
        for _ in tqdm(range(n_simulations), disable=not verbose):
            split_df = self._split(df)
            yield self.analysis.get_standard_error(split_df)

    def _normal_power_calculation(
        self, alpha: float, std_error: float, average_effect: float
    ) -> float:
        """Returns the power of the analysis using the normal distribution.
        Arguments:
            alpha: significance level
            std_error: standard error of the analysis
            average_effect: effect size of the analysis
        """
        if HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.LESS:
            z_alpha = norm.ppf(alpha)
            return float(norm.cdf(z_alpha - average_effect / std_error))

        if HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.GREATER:
            z_alpha = norm.ppf(1 - alpha)
            return 1 - float(norm.cdf(z_alpha - average_effect / std_error))

        if HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
            norm_cdf_right = norm.cdf(z_alpha - average_effect / std_error)
            norm_cdf_left = norm.cdf(-z_alpha - average_effect / std_error)
            return float(norm_cdf_left + (1 - norm_cdf_right))

        raise ValueError(f"{self.analysis.hypothesis} is not a valid HypothesisEntries")

    def _normal_mde_calculation(
        self, alpha: float, std_error: float, power: float
    ) -> float:
        """
        Returns the minimum detectable effect of the analysis using the normal distribution.
        Args:
            alpha: Significance level.
            std_error: Standard error of the analysis.
            power: Power of the analysis.
        """
        if HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.LESS:
            z_alpha = norm.ppf(alpha)
            z_beta = norm.ppf(1 - power)
        elif HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.GREATER:
            z_alpha = norm.ppf(1 - alpha)
            z_beta = norm.ppf(power)
        elif HypothesisEntries(self.analysis.hypothesis) == HypothesisEntries.TWO_SIDED:
            # we are neglecting norm_cdf_left
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(power)
        else:
            raise ValueError(
                f"{self.analysis.hypothesis} is not a valid HypothesisEntries"
            )

        return float(z_alpha + z_beta) * std_error
    
    def _get_time_col(self) -> str:
        if self.time_col is None:
            raise ValueError(
                "Time column not specified. You must provide `time_col` when initializing NormalPowerAnalysis."
            )
        return self.time_col

    def mde_power_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        powers: Iterable[float] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Dict[float, float]:
        """
        Returns the minimum detectable effect of the analysis.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            power: Power of the analysis.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        alpha = self.alpha if alpha is None else alpha
        std_error = self._get_average_standard_error(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            n_simulations=n_simulations,
        )
        return {
            power: self._normal_mde_calculation(
                alpha=alpha, std_error=std_error, power=power
            )
            for power in powers
        }

    def mde(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        power: float = 0.8,
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Returns the minimum detectable effect of the analysis.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            power: Power of the analysis.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        return self.mde_power_line(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            powers=[power],
            n_simulations=n_simulations,
            alpha=alpha,
        )[power]

    def _get_average_standard_error(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        n_simulations: Optional[int] = None,
    ) -> float:
        """
        Gets standard error to be used in normal power calculation.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effects: Average effects to test.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        n_simulations = self.n_simulations if n_simulations is None else n_simulations

        df = df.copy()
        df = self.cupac_handler.add_covariates(df, pre_experiment_df)

        std_errors = list(self._get_standard_error(df, n_simulations, verbose))
        std_error_mean = float(np.mean(std_errors))

        return std_error_mean

    def run_average_standard_error(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        n_simulations: Optional[int] = None,
        experiment_length: Iterable[int] = (),
    ) -> Generator[Tuple[float, int], None, None]:
        """
        Run power analysis by simulation, using standard errors from the analysis.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            n_simulations: Number of simulations to run.
            experiment_length: Length of the experiment in days.
        """
        n_simulations = self.n_simulations if n_simulations is None else n_simulations
        time_col = self._get_time_col()

        for n_days in experiment_length:
            df_time = df.copy()
            experiment_start = df_time[time_col].min()
            df_time = df_time.loc[
                df_time[time_col] < experiment_start + pd.Timedelta(days=n_days)
            ]
            std_error_mean = self._get_average_standard_error(
                df=df_time,
                pre_experiment_df=pre_experiment_df,
                verbose=verbose,
                n_simulations=n_simulations,
            )
            yield std_error_mean, n_days

    def power_time_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effects: Iterable[float] = (),
        experiment_length: Iterable[int] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run power analysis by simulation, using standard errors from the analysis.

        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effects: Average effects to test.
            experiment_length: Length of the experiment in days.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        alpha = self.alpha if alpha is None else alpha

        results = []
        for std_error_mean, n_days in self.run_average_standard_error(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            n_simulations=n_simulations,
            experiment_length=experiment_length,
        ):
            for effect in average_effects:
                power = self._normal_power_calculation(
                    alpha=alpha, std_error=std_error_mean, average_effect=effect
                )
                results.append(
                    {"effect": effect, "power": power, "experiment_length": n_days}
                )

        return results

    def mde_time_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        powers: Iterable[float] = (),
        experiment_length: Iterable[int] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> List[Dict]:
        alpha = self.alpha if alpha is None else alpha

        results = []
        for std_error_mean, n_days in self.run_average_standard_error(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            n_simulations=n_simulations,
            experiment_length=experiment_length,
        ):
            for power in powers:
                mde = self._normal_mde_calculation(
                    alpha=alpha, std_error=std_error_mean, power=power
                )
                results.append(
                    {"power": power, "mde": mde, "experiment_length": n_days}
                )
        return results

    def mde_rolling_time_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        powers: Iterable[float] = (),
        experiment_length: Iterable[int] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
        agg_func: Optional[
            Literal[
                "sum", 
                "mean", 
                "median", 
                "min", 
                "max",
                "count", 
                "std", 
                "var", 
                "nunique", 
                "first", 
                "last",
            ]
        ] = None,
        post_process_func: Optional[Callable[[float], float]] = None,
    ) -> List[Dict]:
        """
        Computes the Minimum Detectable Effect (MDE) for varying experiment lengths
        using a sliding time window, with optional element-wise post-processing 
        on the aggregated metric.

        Args:
            df: Input DataFrame.
            pre_experiment_df: Optional pre-experiment DataFrame.
            powers: Iterable of powers for MDE computation (e.g., [0.8, 0.9]).
            experiment_length: Iterable of experiment durations in days.
            n_simulations: Number of simulations to run (default = self.n_simulations).
            alpha: Significance level (default = self.alpha).
            agg_func: Aggregation function applied to the metric in each cluster window.
            post_process_func: Optional callable applied element-wise to the aggregated metric
                            (like `Series.apply`). Must take a single scalar as input
                            and return a scalar.

        Example with post_process_func:
            def flag_positive(x):
                return 1 if x > 0 else 0

            results = pw.mde_sliding_time_line(
                df=df,
                pre_experiment_df=None,
                time_col="date",
                powers=[0.8],
                experiment_length=[7, 14, 21],
                n_simulations=5,
                agg_func="sum",
                post_process_func=flag_positive
            )

            print(results)
        """
        time_col = self._get_time_col()

        if agg_func is None:
            raise ValueError(
                "Aggregation function `agg_func` must be specified. "
                "Choose one of: 'sum', 'mean', 'median', 'min', 'max', 'count', 'std', 'var', 'nunique', 'first', 'last'."
            )

        alpha = self.alpha if alpha is None else alpha
        n_simulations = self.n_simulations if n_simulations is None else n_simulations
        cluster_cols = self.splitter.cluster_cols
        results = []

        experiment_start = df[time_col].min()

        for n_days in experiment_length:
            df_time = df[df[time_col] <= experiment_start + pd.Timedelta(days=n_days)]

            df_grouped = df_time.groupby(
                cluster_cols, 
                as_index=False
            )[self.target_col].agg(agg_func)

            if post_process_func is not None:
                df_grouped[self.target_col] = df_grouped[self.target_col].apply(post_process_func)

            std_error_mean = self._get_average_standard_error(
                df=df_grouped,
                pre_experiment_df=pre_experiment_df,
                n_simulations=n_simulations,
            )

            for power in powers:
                mde_value = self._normal_mde_calculation(
                    alpha=alpha, std_error=std_error_mean, power=power
                )
                results.append({
                    "power": power,
                    "mde": mde_value,
                    "experiment_length": n_days,
                    "aggregation": agg_func,
                })

        return results

    def power_line(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effects: Iterable[float] = (),
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Dict[float, float]:
        """
        Run power analysis by simulation, using standard errors from the analysis.
        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effects: Average effects to test.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        alpha = self.alpha if alpha is None else alpha

        std_error_mean = self._get_average_standard_error(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            n_simulations=n_simulations,
        )

        return {
            effect: self._normal_power_calculation(
                alpha=alpha, std_error=std_error_mean, average_effect=effect
            )
            for effect in average_effects
        }

    def power_analysis(
        self,
        df: pd.DataFrame,
        pre_experiment_df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        average_effect: float = 0.0,
        n_simulations: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> float:
        """
        Run power analysis by simulation, using standard errors from the analysis.
        Args:
            df: Dataframe with outcome and treatment variables.
            pre_experiment_df: Dataframe with pre-experiment data.
            verbose: Whether to show progress bar.
            average_effect: Average effect of treatment.
            n_simulations: Number of simulations to run.
            alpha: Significance level.
        """
        return self.power_line(
            df=df,
            pre_experiment_df=pre_experiment_df,
            verbose=verbose,
            average_effects=[average_effect],
            n_simulations=n_simulations,
            alpha=alpha,
        )[average_effect]

    def log_nulls(self, df: pd.DataFrame) -> None:
        """Warns about dropping nulls in treatment column"""
        n_nulls = len(df.query(f"{self.treatment_col}.isnull()", engine="python"))
        if n_nulls > 0:
            logging.warning(
                f"There are {n_nulls} null values in treatment, dropping them"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NormalPowerAnalysis":
        """Constructs PowerAnalysis from dictionary"""
        config = PowerConfig(**config_dict)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: PowerConfig) -> "NormalPowerAnalysis":
        """Constructs PowerAnalysis from PowerConfig"""
        splitter_cls = _get_mapping_key(splitter_mapping, config.splitter)
        analysis_cls = _get_mapping_key(analysis_mapping, config.analysis)
        cupac_cls = _get_mapping_key(cupac_model_mapping, config.cupac_model)
        return cls(
            splitter=splitter_cls.from_config(config),
            analysis=analysis_cls.from_config(config),
            cupac_model=cupac_cls.from_config(config),
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            control=config.control,
            n_simulations=config.n_simulations,
            alpha=config.alpha,
            features_cupac_model=config.features_cupac_model,
            seed=config.seed,
            hypothesis=config.hypothesis,
            time_col=config.time_col,
            scale_col=config.scale_col,
        )

    def check_treatment_col(self):
        """Checks consistency of treatment column"""
        assert (
            self.analysis.treatment_col == self.treatment_col
        ), f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in PowerAnalysis ({self.treatment_col})"

        assert (
            self.analysis.treatment_col == self.splitter.treatment_col
        ), f"treatment_col in analysis ({self.analysis.treatment_col}) must be the same as treatment_col in splitter ({self.splitter.treatment_col})"

    def check_target_col(self):
        assert (
            self.analysis.target_col == self.target_col
        ), f"target_col in analysis ({self.analysis.target_col}) must be the same as target_col in PowerAnalysis ({self.target_col})"

    def check_treatment(self):
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
            or isinstance(self.splitter, RepeatedSampler)
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

    def check_scale_col(self):
        if self.scale_col is not None:
            if not isinstance(self.analysis, DeltaMethodAnalysis):
                raise ValueError(
                    "If scale_col is provided, the analysis method must be DeltaMethodAnalysis, since it is the only one that supports scale_col."
                )

    def check_inputs(self):
        self.check_covariates()
        self.check_treatment_col()
        self.check_target_col()
        self.check_treatment()
        self.check_clusters()
        self.check_scale_col()
