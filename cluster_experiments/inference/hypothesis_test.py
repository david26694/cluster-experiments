import copy
from typing import Dict, List, Optional

import pandas as pd

from cluster_experiments.cupac import CupacHandler
from cluster_experiments.experiment_analysis import ExperimentAnalysis, InferenceResults
from cluster_experiments.inference.analysis_results import AnalysisPlanResults
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
    analysis_type : str
        string mapping to an ExperimentAnalysis class. Must be either in the built-in analysis_mapping or in the custom_analysis_type_mapper if provided.
    analysis_config : Optional[dict]
        An optional dictionary representing the configuration for the analysis
    dimensions : Optional[List[Dimension]]
        An optional list of Dimension instances
    cupac_config : Optional[dict]
        An optional dictionary representing the configuration for the cupac model
    custom_analysis_type_mapper : Optional[Dict[str, ExperimentAnalysis]]
        An optional dictionary mapping the names of custom analysis types to the corresponding ExperimentAnalysis classes
    """

    def __init__(
        self,
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict] = None,
        dimensions: Optional[List[Dimension]] = None,
        cupac_config: Optional[dict] = None,
        custom_analysis_type_mapper: Optional[Dict[str, ExperimentAnalysis]] = None,
    ):
        """
        Parameters
        ----------
        metric : Metric
            An instance of the Metric class
        analysis_type : str
            string mapping to an ExperimentAnalysis class. Must be either in the built-in analysis_mapping or in the custom_analysis_type_mapper if provided.
        analysis_config : Optional[dict]
            An optional dictionary representing the configuration for the analysis
        dimensions : Optional[List[Dimension]]
            An optional list of Dimension instances
        cupac_config : Optional[dict]
            An optional dictionary representing the configuration for the cupac model
        custom_analysis_type_mapper : Optional[Dict[str, ExperimentAnalysis]]
            An optional dictionary mapping the names of custom analysis types to the corresponding ExperimentAnalysis classes
        """
        self._validate_inputs(
            metric,
            analysis_type,
            analysis_config,
            dimensions,
            cupac_config,
            custom_analysis_type_mapper,
        )
        self.metric = metric
        self.analysis_type = analysis_type
        self.analysis_config = analysis_config or {}
        self.dimensions = [DefaultDimension()] + (dimensions or [])
        self.cupac_config = cupac_config or {}
        self.custom_analysis_type_mapper = custom_analysis_type_mapper or {}

        self.analysis_type_mapper = self.custom_analysis_type_mapper or analysis_mapping
        self.analysis_class = self.analysis_type_mapper[self.analysis_type]
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
        custom_analysis_type_mapper: Optional[Dict[str, ExperimentAnalysis]] = None,
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
        custom_analysis_type_mapper : Optional[dict[str, ExperimentAnalysis]]
            An optional dictionary mapping the names of custom analysis types to the corresponding ExperimentAnalysis classes
        """
        # Check if metric is a valid Metric instance
        if not isinstance(metric, Metric):
            raise TypeError("Metric must be an instance of Metric")

        # Check if analysis_type is a string
        if not isinstance(analysis_type, str):
            raise TypeError("Analysis must be a string")

        # Check if analysis_config is a dictionary when provided
        if analysis_config is not None and not isinstance(analysis_config, dict):
            raise TypeError("analysis_config must be a dictionary if provided")

        # Check if cupac_config is a dictionary when provided
        if cupac_config is not None and not isinstance(cupac_config, dict):
            raise TypeError("cupac_config must be a dictionary if provided")

        # Check if dimensions is a list of Dimension instances when provided
        if dimensions is not None and (
            not isinstance(dimensions, list)
            or not all(isinstance(dim, Dimension) for dim in dimensions)
        ):
            raise TypeError(
                "Dimensions must be a list of Dimension instances if provided"
            )

        # Validate custom_analysis_type_mapper if provided
        if custom_analysis_type_mapper:
            # Ensure it's a dictionary
            if not isinstance(custom_analysis_type_mapper, dict):
                raise TypeError(
                    "custom_analysis_type_mapper must be a dictionary if provided"
                )

            # Ensure all keys are strings and values are ExperimentAnalysis classes
            for key, value in custom_analysis_type_mapper.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Key '{key}' in custom_analysis_type_mapper must be a string"
                    )
                if not issubclass(value, ExperimentAnalysis):
                    raise TypeError(
                        f"Value '{value}' for key '{key}' in custom_analysis_type_mapper must be a subclass of ExperimentAnalysis"
                    )

            # Ensure the analysis_type is in the custom mapper if a custom mapper is provided
            if analysis_type not in custom_analysis_type_mapper:
                raise ValueError(
                    f"Analysis type '{analysis_type}' not found in the provided custom_analysis_type_mapper"
                )

        # If no custom_analysis_type_mapper, check if analysis_type exists in the default mapping
        elif analysis_type not in analysis_mapping:
            raise ValueError(
                f"Analysis type '{analysis_type}' not found in analysis_mapping"
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

    def cupac_not_in_covariates(self, covariates: List[str]) -> None:
        """
        Checks if any cupac covariate is missing from the covariates provided in the analysis_config.
        """
        return any([col not in covariates for col in self.cupac_covariate_col])

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

        if self.cupac_covariate_col and self.cupac_not_in_covariates(covariates):
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

    def add_covariates(
        self, exp_data: pd.DataFrame, pre_exp_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        If the test is a cupac test, adds the covariates to the experimental data.
        """
        if self.is_cupac:
            exp_data = self.cupac_handler.add_covariates(
                df=exp_data, pre_experiment_df=pre_exp_data
            )

        return exp_data

    def get_test_results(
        self,
        control_variant: Variant,
        treatment_variant: Variant,
        variant_col: str,
        exp_data: pd.DataFrame,
        dimension: Dimension,
        dimension_value: str,
        alpha: float,
    ) -> AnalysisPlanResults:
        """
        Performs the hypothesis test on the provided data, for the given dimension value.

        Parameters
        ----------
        control_variant : Variant
            The control variant
        treatment_variant : Variant
            The treatment variant
        variant_col : str
            The column name representing the variant
        exp_data : pd.DataFrame
            The dataframe containing the data for analysis.
        dimension : Dimension
            The dimension instance
        dimension_value : str
            The value of the dimension
        alpha : float
            The significance level to be used in the inference analysis.

        Returns
        -------
        AnalysisPlanResults
            The results of the hypothesis test
        """
        self._prepare_analysis_config(
            target_col=self.metric.target_column,
            treatment_col=variant_col,
            treatment=treatment_variant.name,
        )

        prepared_df = self.prepare_data(
            data=exp_data,
            variant_col=variant_col,
            treatment_variant=treatment_variant,
            control_variant=control_variant,
            dimension_name=dimension.name,
            dimension_value=dimension_value,
        )

        inference_results = self.get_inference_results(df=prepared_df, alpha=alpha)

        control_variant_mean = self.metric.get_mean(
            prepared_df.query(f"{variant_col}=='{control_variant.name}'")
        )
        treatment_variant_mean = self.metric.get_mean(
            prepared_df.query(f"{variant_col}=='{treatment_variant.name}'")
        )

        test_results = AnalysisPlanResults(
            metric_alias=[self.metric.alias],
            control_variant_name=[control_variant.name],
            treatment_variant_name=[treatment_variant.name],
            control_variant_mean=[control_variant_mean],
            treatment_variant_mean=[treatment_variant_mean],
            analysis_type=[self.analysis_type],
            ate=[inference_results.ate],
            ate_ci_lower=[inference_results.conf_int.lower],
            ate_ci_upper=[inference_results.conf_int.upper],
            p_value=[inference_results.p_value],
            std_error=[inference_results.std_error],
            dimension_name=[dimension.name],
            dimension_value=[dimension_value],
            alpha=[alpha],
        )

        return test_results
