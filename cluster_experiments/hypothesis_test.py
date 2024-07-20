from typing import List, Optional

from cluster_experiments.dimension import Dimension
from cluster_experiments.metric import Metric


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

    Methods
    -------
    __init__(self, metric: Metric, analysis: ExperimentAnalysis, analysis_config: Optional[dict] = None, dimensions: Optional[List[Dimension]] = None):
        Initializes the HypothesisTest with the provided metric, analysis, and optional analysis configuration and dimensions.
    _validate_inputs(metric: Metric, analysis: ExperimentAnalysis, analysis_config: Optional[dict], dimensions: Optional[List[Dimension]]):
        Validates the inputs for the HypothesisTest class.
    """

    def __init__(
        self,
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict] = None,
        dimensions: Optional[List[Dimension]] = None,
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
        """
        self._validate_inputs(metric, analysis_type, analysis_config, dimensions)
        self.metric = metric
        self.analysis_type = analysis_type
        self.analysis_config = analysis_config or {}
        self.dimensions = dimensions or [Dimension(name="total", values=["total"])]

    @staticmethod
    def _validate_inputs(
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict],
        dimensions: Optional[List[Dimension]],
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

        Raises
        ------
        TypeError
            If metric is not an instance of Metric, if analysis_type is not an instance of string,
            if analysis_config is not a dictionary (when provided), or if dimensions is not a list of Dimension instances (when provided).
        """
        if not isinstance(metric, Metric):
            raise TypeError("Metric must be an instance of Metric")
        if not isinstance(analysis_type, str):
            raise TypeError("Analysis must be a string")
        # todo: add better check for analysis_type allowed values
        if analysis_config is not None and not isinstance(analysis_config, dict):
            raise TypeError("Analysis_config must be a dictionary if provided")
        if dimensions is not None and (
            not isinstance(dimensions, list)
            or not all(isinstance(dim, Dimension) for dim in dimensions)
        ):
            raise TypeError(
                "Dimensions must be a list of Dimension instances if provided"
            )
