from dataclasses import dataclass
from typing import Dict, List, Optional

from cluster_experiments.experiment_analysis import (
    GeeExperimentAnalysis,
    GeeExperimentAnalysisAggMean,
)
from cluster_experiments.perturbator import BinaryPerturbator, UniformPerturbator
from cluster_experiments.pre_experiment_covariates import (
    PreExperimentFeaturizer,
    TargetAggregation,
)
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    SwitchbackSplitter,
)


@dataclass
class PowerConfig:
    """
    Dataclass to create a power analysis from.
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
    clusters: List[str]
    cluster_cols: List[str]

    # optional mappings
    featurizer: str = ""

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

    # PreExperimentFeaturizer
    agg_col: str = ""
    smoothing_factor: float = 20


perturbator_mapping = {
    "binary": BinaryPerturbator,
    "uniform": UniformPerturbator,
}

splitter_mapping = {
    "clustered": ClusteredSplitter,
    "clustered_balance": BalancedClusteredSplitter,
    "switchback": SwitchbackSplitter,
    "switchback_balance": BalancedSwitchbackSplitter,
}

analysis_mapping = {
    "gee": GeeExperimentAnalysis,
    "gee_mean": GeeExperimentAnalysisAggMean,
}

featurizer_mapping = {"": PreExperimentFeaturizer, "mean_featurizer": TargetAggregation}
