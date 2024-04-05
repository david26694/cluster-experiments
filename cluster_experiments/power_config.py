import datetime
import logging
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel

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
    BetaRelativePerturbator,
    BetaRelativePositivePerturbator,
    BinaryPerturbator,
    ConstantPerturbator,
    NormalPerturbator,
    RelativePositivePerturbator,
    SegmentedBetaRelativePerturbator,
    UniformPerturbator,
)
from cluster_experiments.random_splitter import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    ClusteredSplitter,
    NonClusteredSplitter,
    RepeatedSampler,
    StratifiedClusteredSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
)


class PerturbatorEnum(Enum):
    BINARY = "binary"
    CONSTANT = "constant"
    UNIFORM = "uniform"
    RELATIVE_POSITIVE = "relative_positive"
    NORMAL = "normal"
    BETA_RELATIVE_POSITIVE = "beta_relative_positive"
    BETA_RELATIVE = "beta_relative"
    SEGMENTED_BETA_RELATIVE = "segmented_beta_relative"


class SplitterEnum(Enum):
    CLUSTERED = "clustered"
    CLUSTERED_BALANCE = "clustered_balance"
    CLUSTERED_BALANCED = "clustered_balanced"
    BALANCED_CLUSTER = "balanced_cluster"
    BALANCED_CLUSTERED = "balanced_clustered"
    NON_CLUSTERED = "non_clustered"
    CLUSTERED_STRATIFIED = "clustered_stratified"
    STRATIFIED_CLUSTER = "stratified_cluster"
    STRATIFIED_CLUSTERED = "stratified_clustered"
    SWITCHBACK = "switchback"
    SWITCHBACK_BALANCE = "switchback_balance"
    SWITCHBACK_BALANCED = "switchback_balanced"
    BALANCED_SWITCHBACK = "balanced_switchback"
    SWITCHBACK_STRATIFIED = "switchback_stratified"
    STRATIFIED_SWITCHBACK = "stratified_switchback"
    REPEATED_SAMPLER = "repeated_sampler"


class AnalysisEnum(Enum):
    GEE = "gee"
    OLS_NON_CLUSTERED = "ols_non_clustered"
    OLS = "ols"
    OLS_CLUSTERED = "ols_clustered"
    CLUSTERED_OLS = "clustered_ols"
    TTEST_CLUSTERED = "ttest_clustered"
    PAIRED_TTEST_CLUSTERED = "paired_ttest_clustered"
    MLM = "mlm"


class CupacModelEnum(Enum):
    EMPTY_REGRESSOR = ""
    MEAN_CUPAC_MODEL = "mean_cupac_model"


perturbator_mapping = {
    PerturbatorEnum.BINARY: BinaryPerturbator,
    PerturbatorEnum.CONSTANT: ConstantPerturbator,
    PerturbatorEnum.UNIFORM: UniformPerturbator,
    PerturbatorEnum.RELATIVE_POSITIVE: RelativePositivePerturbator,
    PerturbatorEnum.NORMAL: NormalPerturbator,
    PerturbatorEnum.BETA_RELATIVE_POSITIVE: BetaRelativePositivePerturbator,
    PerturbatorEnum.BETA_RELATIVE: BetaRelativePerturbator,
    PerturbatorEnum.SEGMENTED_BETA_RELATIVE: SegmentedBetaRelativePerturbator,
}

splitter_mapping = {
    SplitterEnum.CLUSTERED: ClusteredSplitter,
    SplitterEnum.CLUSTERED_BALANCE: BalancedClusteredSplitter,
    SplitterEnum.CLUSTERED_BALANCED: BalancedClusteredSplitter,
    SplitterEnum.BALANCED_CLUSTER: BalancedClusteredSplitter,
    SplitterEnum.BALANCED_CLUSTERED: BalancedClusteredSplitter,
    SplitterEnum.NON_CLUSTERED: NonClusteredSplitter,
    SplitterEnum.CLUSTERED_STRATIFIED: StratifiedClusteredSplitter,
    SplitterEnum.STRATIFIED_CLUSTER: StratifiedClusteredSplitter,
    SplitterEnum.STRATIFIED_CLUSTERED: StratifiedClusteredSplitter,
    SplitterEnum.SWITCHBACK: SwitchbackSplitter,
    SplitterEnum.SWITCHBACK_BALANCE: BalancedSwitchbackSplitter,
    SplitterEnum.SWITCHBACK_BALANCED: BalancedSwitchbackSplitter,
    SplitterEnum.BALANCED_SWITCHBACK: BalancedSwitchbackSplitter,
    SplitterEnum.SWITCHBACK_STRATIFIED: StratifiedSwitchbackSplitter,
    SplitterEnum.STRATIFIED_SWITCHBACK: StratifiedSwitchbackSplitter,
    SplitterEnum.REPEATED_SAMPLER: RepeatedSampler,
}

analysis_mapping = {
    AnalysisEnum.GEE: GeeExperimentAnalysis,
    AnalysisEnum.OLS_NON_CLUSTERED: OLSAnalysis,
    AnalysisEnum.OLS: OLSAnalysis,
    AnalysisEnum.OLS_CLUSTERED: ClusteredOLSAnalysis,
    AnalysisEnum.CLUSTERED_OLS: ClusteredOLSAnalysis,
    AnalysisEnum.TTEST_CLUSTERED: TTestClusteredAnalysis,
    AnalysisEnum.PAIRED_TTEST_CLUSTERED: PairedTTestClusteredAnalysis,
    AnalysisEnum.MLM: MLMExperimentAnalysis,
}


cupac_model_mapping = {
    CupacModelEnum.EMPTY_REGRESSOR: EmptyRegressor,
    CupacModelEnum.MEAN_CUPAC_MODEL: TargetAggregation,
}


class MissingArgumentError(ValueError):
    pass


class PowerConfig(BaseModel):
    """
    Dataclass to create a power analysis from.

        splitter: Splitter object to use
        perturbator: Perturbator object to use
        analysis: ExperimentAnalysis object to use
        washover: Washover object to use, defaults to ""
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
        range_min: minimum value of the target range for relative beta perturbator, must be >-1
        range_max: maximum value of the target range for relative beta perturbator
        reduce_variance: whether to reduce variance in the BetaRelative perturbator
        segment_cols: list of segmentation columns for segmented perturbator
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
        perturbator="constant",
        cluster_cols=["city"],
        n_simulations=100,
        alpha=0.05,
    )
    power_analysis = PowerAnalysis.from_config(p)
    ```
    """

    # mappings
    perturbator: PerturbatorEnum
    splitter: SplitterEnum
    analysis: AnalysisEnum
    washover: str = ""

    # Needed
    cluster_cols: Optional[List[str]] = None

    # optional mappings
    cupac_model: CupacModelEnum = CupacModelEnum.EMPTY_REGRESSOR

    # Shared
    target_col: str = "target"
    treatment_col: str = "treatment"
    treatment: str = "B"

    # Perturbator
    average_effect: Optional[float] = None
    scale: Optional[float] = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    reduce_variance: Optional[bool] = None
    segment_cols: Optional[List[str]] = None

    # Splitter
    treatments: Optional[List[str]] = None
    strata_cols: Optional[List[str]] = None
    splitter_weights: Optional[List[float]] = None
    switch_frequency: Optional[str] = None
    # Switchback
    time_col: Optional[str] = None
    washover_time_delta: Optional[Union[datetime.timedelta, int]] = None

    # Analysis
    covariates: Optional[List[str]] = None
    hypothesis: str = "two-sided"

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
            if self._are_different(self.switch_frequency, None):
                self._set_and_log("switch_frequency", None, "splitter")
            if self._are_different(self.washover_time_delta, None):
                self._set_and_log("washover_time_delta", None, "splitter")
            if self._are_different(self.washover, ""):
                self._set_and_log("washover", "", "splitter")
            if self._are_different(self.time_col, None):
                self._set_and_log("time_col", None, "splitter")

        if self.perturbator not in {"normal", "beta_relative_positive"}:
            if self._are_different(self.scale, None):
                self._set_and_log("scale", None, "perturbator")

        if self.perturbator not in {"beta_relative", "segmented_beta_relative"}:
            if self._are_different(self.range_min, None):
                self._set_and_log("range_min", None, "perturbator")
            if self._are_different(self.range_max, None):
                self._set_and_log("range_max", None, "perturbator")
            if self._are_different(self.reduce_variance, None):
                self._set_and_log("reduce_variance", None, "perturbator")

        if self.perturbator not in {"segmented_beta_relative"}:
            if self._are_different(self.segment_cols, None):
                self._set_and_log("segment_cols", None, "perturbator")

        if "stratified" not in self.splitter and "paired_ttest" not in self.analysis:
            if self._are_different(self.strata_cols, None):
                self._set_and_log("strata_cols", None, "splitter")

        if "stratified" in self.splitter or "balanced" in self.splitter:
            if self._are_different(self.splitter_weights, None):
                self._set_and_log("splitter_weights", None, "splitter")

        if self.cupac_model != "mean_cupac_model":
            if self._are_different(self.agg_col, ""):
                self._set_and_log("agg_col", "", "cupac_model")
            if self._are_different(self.smoothing_factor, 20):
                self._set_and_log("smoothing_factor", 20, "cupac_model")
        # for now, features_cupac_model are not used
        # TODO: raise loudly when features_cupac_model is not None
        if self._are_different(self.features_cupac_model, None):
            self._set_and_log("features_cupac_model", None, "cupac_model")

        if "ttest" in self.analysis:
            if self._are_different(self.covariates, None):
                self._set_and_log("covariates", None, "analysis")

        if "segmented" in self.perturbator:
            self._raise_error_if_missing("segment_cols", "perturbator")

    def _are_different(self, arg1, arg2) -> bool:
        return arg1 != arg2

    def _set_and_log(self, attr, value, other_attr):
        logging.warning(
            f"{attr} = {getattr(self, attr)} has no effect with "
            f"{other_attr} = {getattr(self, other_attr)}. "
            f"Overriding {attr} to {value}."
        )
        setattr(self, attr, value)

    def _raise_error_if_missing(self, attr, other_attr):
        if getattr(self, attr) is None:
            raise MissingArgumentError(
                f"{attr} is required when using "
                f"{other_attr} = {getattr(self, other_attr)}."
            )
