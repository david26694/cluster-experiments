import copy
from typing import List, Optional

import pandas as pd

from cluster_experiments.cupac import CupacHandler
from cluster_experiments.experiment_analysis import InferenceResults
from cluster_experiments.inference.dimension import DefaultDimension, Dimension
from cluster_experiments.inference.metric import Metric
from cluster_experiments.inference.variant import Variant
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
        self.cupac_handler = (
            CupacHandler(**self.cupac_config) if self.is_cupac else None
        )
        self.cupac_covariate_col = (
            self.cupac_handler.cupac_outcome_name if self.is_cupac else None
        )

        self.new_analysis_config = None
        self.experiment_analysis = None

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

        self.experiment_analysis = self.analysis_class(**self.new_analysis_config)
        inference_results = self.experiment_analysis.get_inference_results(
            df=df, alpha=alpha
        )

        return inference_results

    def _prepare_analysis_config(
        self, target_col: str, treatment_col: str, treatment: str
    ) -> None:
        """
        Extends the analysis_config provided by the user, by adding or overriding the following keys:
        - target_col
        - treatment_col
        - treatment

        Also handles cupac covariate.

        Returns
        -------
        dict
            The prepared analysis configuration, ready to be ingested by the experiment analysis class
        """
        new_analysis_config = copy.deepcopy(self.analysis_config)

        new_analysis_config["target_col"] = target_col
        new_analysis_config["treatment_col"] = treatment_col
        new_analysis_config["treatment"] = treatment

        covariates = new_analysis_config.get("covariates", [])

        if self.cupac_covariate_col and self.cupac_covariate_col not in covariates:
            raise ValueError(
                f"You provided a cupac configuration but did not provide the cupac covariate called {self.cupac_covariate_col} in the analysis_config"
            )

        self.new_analysis_config = new_analysis_config

    @staticmethod
    def prepare_data(
        data: pd.DataFrame,
        variant_col: str,
        treatment_variant: Variant,
        control_variant: Variant,
        dimension_name: str,
        dimension_value: str,
    ) -> pd.DataFrame:
        """
        Prepares the data for the experiment analysis pipeline
        """
        prepared_df = data.copy()

        prepared_df = prepared_df.assign(__total_dimension="total")

        prepared_df = prepared_df.query(
            f"{variant_col}.isin(['{treatment_variant.name}','{control_variant.name}'])"
        ).query(f"{dimension_name} == '{dimension_value}'")

        return prepared_df
