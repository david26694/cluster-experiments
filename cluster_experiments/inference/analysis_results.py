from dataclasses import asdict, dataclass, field
from typing import List

import pandas as pd


@dataclass
class AnalysisPlanResults:
    """
    A dataclass used to represent the results of the experiment analysis.

    Attributes
    ----------
    metric_alias : List[str]
        The alias of the metric used in the test
    control_variant_name : List[str]
        The name of the control variant
    treatment_variant_name : List[str]
        The name of the treatment variant
    control_variant_mean : List[float]
        The mean value of the control variant
    treatment_variant_mean : List[float]
        The mean value of the treatment variant
    analysis_type : List[str]
        The type of analysis performed
    ate : List[float]
        The average treatment effect
    ate_ci_lower : List[float]
        The lower bound of the confidence interval for the ATE
    ate_ci_upper : List[float]
        The upper bound of the confidence interval for the ATE
    p_value : List[float]
        The p-value of the test
    std_error : List[float]
        The standard error of the test
    dimension_name : List[str]
        The name of the dimension
    dimension_value : List[str]
        The value of the dimension
    alpha: List[float]
        The significance level of the test
    """

    metric_alias: List[str] = field(default_factory=lambda: [])
    control_variant_name: List[str] = field(default_factory=lambda: [])
    treatment_variant_name: List[str] = field(default_factory=lambda: [])
    control_variant_mean: List[float] = field(default_factory=lambda: [])
    treatment_variant_mean: List[float] = field(default_factory=lambda: [])
    analysis_type: List[str] = field(default_factory=lambda: [])
    ate: List[float] = field(default_factory=lambda: [])
    ate_ci_lower: List[float] = field(default_factory=lambda: [])
    ate_ci_upper: List[float] = field(default_factory=lambda: [])
    p_value: List[float] = field(default_factory=lambda: [])
    std_error: List[float] = field(default_factory=lambda: [])
    dimension_name: List[str] = field(default_factory=lambda: [])
    dimension_value: List[str] = field(default_factory=lambda: [])
    alpha: List[float] = field(default_factory=lambda: [])

    def __add__(self, other):
        if not isinstance(other, AnalysisPlanResults):
            return NotImplemented

        return AnalysisPlanResults(
            metric_alias=self.metric_alias + other.metric_alias,
            control_variant_name=self.control_variant_name + other.control_variant_name,
            treatment_variant_name=self.treatment_variant_name
            + other.treatment_variant_name,
            control_variant_mean=self.control_variant_mean + other.control_variant_mean,
            treatment_variant_mean=self.treatment_variant_mean
            + other.treatment_variant_mean,
            analysis_type=self.analysis_type + other.analysis_type,
            ate=self.ate + other.ate,
            ate_ci_lower=self.ate_ci_lower + other.ate_ci_lower,
            ate_ci_upper=self.ate_ci_upper + other.ate_ci_upper,
            p_value=self.p_value + other.p_value,
            std_error=self.std_error + other.std_error,
            dimension_name=self.dimension_name + other.dimension_name,
            dimension_value=self.dimension_value + other.dimension_value,
            alpha=self.alpha + other.alpha,
        )

    def to_dataframe(self):
        return pd.DataFrame(asdict(self))
