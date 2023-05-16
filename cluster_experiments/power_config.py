from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import List, Optional

from cluster_experiments.cupac import EmptyRegressor, TargetAggregation
from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    GeeExperimentAnalysis,
    MLMExperimentAnalysis,
    OLSAnalysis,
    PairedTTestClusteredAnalysis,
    TTestClusteredAnalysis,
)
from cluster_experiments.perturbator import (
    BetaRelativePositivePerturbator,
    BinaryPerturbator,
    NormalPerturbator,
    RelativePositivePerturbator,
    UniformPerturbator,
)
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    NonClusteredSplitter,
    StratifiedClusteredSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
)


@dataclass(frozen=True)
class PowerConfig:
    """
    Dataclass to create a power analysis from.

    Arguments:
        splitter: Splitter object to use
        perturbator: Perturbator object to use
        analysis: ExperimentAnalysis object to use
        cupac_model: CUPAC model to use
        n_simulations: number of simulations to run
        cluster_cols: list of columns to use as clusters
        target_col: column to use as target
        treatment_col: column to use as treatment
        treatment: what value of treatment_col should be considered as treatment
        control: what value of treatment_col should be considered as control
        strata_cols: columns to stratify with
        splitter_weights: weights to use for the splitter, should have the same length as treatments, each weight should correspond to an element in treatments
        switch_frequency: how often to switch treatments
        time_col: column to use as time in switchback splitter
        washover_time_delta: optional, int indicating the washover time in minutes or datetime.timedelta object
        covariates: list of columns to use as covariates
        average_effect: average effect to use in the perturbator
        scale: scale to use in stochastic perturbators
        treatments: list of treatments to use
        alpha: alpha value to use in the power analysis
        agg_col: column to use for aggregation in the CUPAC model
        smoothing_factor: smoothing value to use in the CUPAC model
        features_cupac_model: list of features to use in the CUPAC model
        seed: seed to make the power analysis reproducible

    Usage:

    ```python
    from cluster_experiments.power_config import PowerConfig
    from cluster_experiments.power_analysis import PowerAnalysis

    p = PowerConfig(
        analysis="gee",
        splitter="clustered_balance",
        perturbator="uniform",
        cluster_cols=["city"],
        n_simulations=100,
        alpha=0.05,
    )
    power_analysis = PowerAnalysis.from_config(p)
    ```
    """

    # mappings
    perturbator: str
    splitter: str
    analysis: str
    washover: str = ""

    # Needed
    cluster_cols: Optional[List[str]] = None

    # optional mappings
    cupac_model: str = ""

    # Shared
    target_col: str = "target"
    treatment_col: str = "treatment"
    treatment: str = "B"

    # Perturbator
    average_effect: Optional[float] = None
    scale: Optional[float] = None

    # Splitter
    treatments: Optional[List[str]] = None
    strata_cols: Optional[List[str]] = None
    splitter_weights: Optional[List[float]] = None
    switch_frequency: Optional[str] = None
    # Switchback
    time_col: Optional[str] = None
    washover_time_delta: Optional[datetime.timedelta | int] = None

    # Analysis
    covariates: Optional[List[str]] = None

    # Power analysis
    n_simulations: int = 100
    alpha: float = 0.05
    control: str = "A"

    # Cupac
    agg_col: str = ""
    smoothing_factor: float = 20
    features_cupac_model: Optional[List[str]] = None

    seed: Optional[int] = None


def __post_init__(self):
    if "switchback" not in self.splitter:
        self.switch_frequency = None
        self.washover_time_delta = None
        self.washover = ""
        self.time_col = None

    if self.perturbator not in {"normal", "beta_relative_positive"}:
        self.scale = None

    if "balanced" not in self.splitter:
        self.treatments = None

    if "stratified" not in self.splitter:
        self.strata_cols = None

    if "cupac" not in self.analysis:
        self.agg_col = ""
        self.smoothing_factor = 20
        self.features_cupac_model = None

    if "ttest" in self.analysis:
        self.covariates = None


perturbator_mapping = {
    "binary": BinaryPerturbator,
    "uniform": UniformPerturbator,
    "relative_positive": RelativePositivePerturbator,
    "normal": NormalPerturbator,
    "beta_relative_positive": BetaRelativePositivePerturbator,
}

splitter_mapping = {
    "clustered": ClusteredSplitter,
    "clustered_balance": BalancedClusteredSplitter,
    "non_clustered": NonClusteredSplitter,
    "clustered_stratified": StratifiedClusteredSplitter,
    "switchback": SwitchbackSplitter,
    "switchback_balance": BalancedSwitchbackSplitter,
    "switchback_stratified": StratifiedSwitchbackSplitter,
}

analysis_mapping = {
    "gee": GeeExperimentAnalysis,
    "ols_non_clustered": OLSAnalysis,
    "ols_clustered": ClusteredOLSAnalysis,
    "ttest_clustered": TTestClusteredAnalysis,
    "paired_ttest_clustered": PairedTTestClusteredAnalysis,
    "mlm": MLMExperimentAnalysis,
}

cupac_model_mapping = {"": EmptyRegressor, "mean_cupac_model": TargetAggregation}
