from typing import List, Optional

import pandas as pd

from cluster_experiments.experiment_analysis import InferenceResults
from cluster_experiments.inference.dimension import DefaultDimension, Dimension
from cluster_experiments.inference.metric import Metric
from cluster_experiments.power_config import analysis_mapping


class HypothesisTest:
    """
    A class used to represent a Hypothesis Test with a metric, analysis, optional analysis configuration, and optional dimensions.

    Attributes
    ----------
    metric : Metric
        An instance of the Metric class
    analysis : ExperimentAnalysis
        An instance of the ExperimentAnalysis class
    analysis_config : Optional[dict]
        An optional dictionary representing the configuration for the analysis
    dimensions : Optional[List[Dimension]]
        An optional list of Dimension instances
    cupac_config : Optional[dict]
            An optional dictionary representing the configuration for the cupac model
    """

    def __init__(
        self,
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict] = None,
        dimensions: Optional[List[Dimension]] = None,
        cupac_config: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        metric : Metric
            An instance of the Metric class
        analysis_type : str
            string mapper to an ExperimentAnalysis
        analysis_config : Optional[dict]
            An optional dictionary representing the configuration for the analysis
        dimensions : Optional[List[Dimension]]
            An optional list of Dimension instances
        cupac_config : Optional[dict]
            An optional dictionary representing the configuration for the cupac model
        """
        self._validate_inputs(metric, analysis_type, analysis_config, dimensions)
        self.metric = metric
        self.analysis_type = analysis_type
        self.analysis_config = analysis_config or {}
        self.dimensions = [DefaultDimension()] + (dimensions or [])
        self.cupac_config = cupac_config or {}

        self.analysis_class = analysis_mapping[self.analysis_type]
        self.is_cupac = bool(cupac_config)

    @staticmethod
    def _validate_inputs(
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict],
        dimensions: Optional[List[Dimension]],
        cupac_config: Optional[dict] = None,
    ):
        """
        Validates the inputs for the HypothesisTest class.

        Parameters
        ----------
        metric : Metric
            An instance of the Metric class
        analysis_type : str
            string mapper to an ExperimentAnalysis
        analysis_config : Optional[dict]
            An optional dictionary representing the configuration for the analysis
        dimensions : Optional[List[Dimension]]
            An optional list of Dimension instances
        cupac_config : Optional[dict]
            An optional dictionary representing the configuration for the cupac model

        Raises
        ------
        TypeError
            If metric is not an instance of Metric, if analysis_type is not an instance of string,
            if analysis_config is not a dictionary (when provided), or if dimensions is not a list of Dimension instances (when provided),
            if cupac_config is not a dictionary (when provided)
        """
        if not isinstance(metric, Metric):
            raise TypeError("Metric must be an instance of Metric")
        if not isinstance(analysis_type, str):
            raise TypeError("Analysis must be a string")
        # todo: add better check for analysis_type allowed values
        if analysis_config is not None and not isinstance(analysis_config, dict):
            raise TypeError("analysis_config must be a dictionary if provided")
        if cupac_config is not None and not isinstance(analysis_config, dict):
            raise TypeError("cupac_config must be a dictionary if provided")
        if dimensions is not None and (
            not isinstance(dimensions, list)
            or not all(isinstance(dim, Dimension) for dim in dimensions)
        ):
            raise TypeError(
                "Dimensions must be a list of Dimension instances if provided"
            )

    def get_inference_results(self, df: pd.DataFrame, alpha: float) -> InferenceResults:
        """
        Performs inference analysis on the provided DataFrame using the analysis class.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data for analysis.
        alpha : float
            The significance level to be used in the inference analysis.

        Returns
        -------
        InferenceResults
            The results containing the statistics of the inference procedure.
        """

        inference_results = self.analysis_class.get_inference_results(
            df=df, alpha=alpha
        )

        return inference_results
