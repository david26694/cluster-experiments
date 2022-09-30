from dataclasses import dataclass
from typing import Dict, List, Optional

from cluster_experiments.cupac import EmptyRegressor, TargetAggregation
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis, OLSAnalysis
from cluster_experiments.perturbator import BinaryPerturbator, UniformPerturbator
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    NonClusteredSplitter,
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
        covariates: list of columns to use as covariates
        clusters: list of clusters to use
        dates: list of dates to use
        average_effect: average effect to use in the perturbator
        treatments: list of treatments to use
        cluster_mapping: mapping of clusters and columns
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
        clusters=["A", "B", "C"],
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
    clusters: Optional[List[str]] = None
    cluster_cols: Optional[List[str]] = None

    # optional mappings
    cupac_model: str = ""

    # Shared
    target_col: str = "target"
    treatment_col: str = "treatment"
    treatment: str = "B"

    # Perturbator
    average_effect: float = 0.0

    # Splitter
    treatments: Optional[List[str]] = None
    dates: Optional[List[str]] = None
    cluster_mapping: Optional[Dict[str, str]] = None

    # Analysis
    covariates: Optional[List[str]] = None

    # Power analysis
    n_simulations: int = 100
    alpha: float = 0.05

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
    "switchback": SwitchbackSplitter,
    "switchback_balance": BalancedSwitchbackSplitter,
    "non_clustered": NonClusteredSplitter,
}

analysis_mapping = {"gee": GeeExperimentAnalysis, "ols_non_clustered": OLSAnalysis}

cupac_model_mapping = {"": EmptyRegressor, "mean_cupac_model": TargetAggregation}
