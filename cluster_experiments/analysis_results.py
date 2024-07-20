from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class HypothesisTestResults:
    """
    A dataclass used to represent the results of a Hypothesis Test.

    Attributes
    ----------
    metric_alias : str
        The alias of the metric used in the test
    control_variant_name : str
        The name of the control variant
    treatment_variant_name : str
        The name of the treatment variant
    control_variant_mean : float
        The mean value of the control variant
    treatment_variant_mean : float
        The mean value of the treatment variant
    analysis_type : str
        The type of analysis performed
    ate : float
        The average treatment effect
    ate_ci_lower : float
        The lower bound of the confidence interval for the ATE
    ate_ci_upper : float
        The upper bound of the confidence interval for the ATE
    p_value : float
        The p-value of the test
    std_error : float
        The standard error of the test
    dimension_name : str
        The name of the dimension
    dimension_value : str
        The value of the dimension
    """

    metric_alias: str
    control_variant_name: str
    treatment_variant_name: str
    control_variant_mean: float
    treatment_variant_mean: float
    analysis_type: str
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    p_value: float
    std_error: float
    dimension_name: str
    dimension_value: str


class AnalysisPlanResults(pd.DataFrame):
    """
    A class used to represent the results of an Analysis Plan as a pandas DataFrame.

    Methods
    -------
    add_results(results: List[HypothesisTestResults]):
        Adds a list of new results to the DataFrame.
    """

    def __init__(self, *args, **kwargs):
        columns = [
            "metric_alias",
            "control_variant_name",
            "treatment_variant_name",
            "control_variant_mean",
            "treatment_variant_mean",
            "analysis_type",
            "ate",
            "ate_ci_lower",
            "ate_ci_upper",
            "p_value",
            "std_error",
            "dimension_name",
            "dimension_value",
        ]
        super().__init__(*args, columns=columns, **kwargs)

    @staticmethod
    def add_results(results: List[HypothesisTestResults]) -> pd.DataFrame:
        """
        Adds a list of new results to the DataFrame.

        Parameters
        ----------
        results : List[HypothesisTestResults]
            The list of results to be added to the DataFrame
        """
        new_data = [result.__dict__ for result in results]
        new_df = pd.DataFrame(new_data)
        return new_df
