from dataclasses import dataclass
from typing import List, Optional

from cluster_experiments.cupac import EmptyRegressor, TargetAggregation
from cluster_experiments.experiment_analysis import (
    ClusteredOLSAnalysis,
    GeeExperimentAnalysis,
    OLSAnalysis,
    TTestClusteredAnalysis,
)
from cluster_experiments.perturbator import BinaryPerturbator, UniformPerturbator
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    NonClusteredSplitter,
    StratifiedClusteredSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
)


@dataclass
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
        covariates: list of columns to use as covariates
        average_effect: average effect to use in the perturbator
        treatments: list of treatments to use
        alpha: alpha value to use in the power analysis
        agg_col: column to use for aggregation in the CUPAC model
        smoothing_factor: smoothing value to use in the CUPAC model
        features_cupac_model: list of features to use in the CUPAC model

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

    # Splitter
    treatments: Optional[List[str]] = None
    strata_cols: Optional[List[str]] = None
    splitter_weights: Optional[List[float]] = None
    switch_frequency: Optional[str] = None
    time_col: Optional[str] = None

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


perturbator_mapping = {
    "binary": BinaryPerturbator,
    "uniform": UniformPerturbator,
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
}

cupac_model_mapping = {"": EmptyRegressor, "mean_cupac_model": TargetAggregation}
