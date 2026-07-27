import copy
from typing import Dict, List, Optional

import pandas as pd

from cluster_experiments.cupac import CupacHandler, MLRateHandler, NoOpHandler
from cluster_experiments.experiment_analysis import ExperimentAnalysis, InferenceResults
from cluster_experiments.inference.analysis_results import AnalysisPlanResults
from cluster_experiments.inference.dimension import DefaultDimension, Dimension
from cluster_experiments.inference.metric import Metric, RatioMetric
from cluster_experiments.inference.split import DefaultSplit, Split
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
        An optional list of Dimension instances. Dimensions describe stable unit attributes.
    splits : Optional[List[Split]]
        An optional list of Split instances. Splits describe attributes that can change during the experiment.
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
        splits: Optional[List[Split]] = None,
        cupac_config: Optional[dict] = None,
        ml_option: str = "cupac",
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
        splits : Optional[List[Split]]
            An optional list of Split instances
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
            splits,
            cupac_config,
            custom_analysis_type_mapper,
        )
        self.metric = metric
        self.analysis_type = analysis_type
        self.analysis_config = analysis_config or {}
        self.dimensions = [DefaultDimension()] + (dimensions or [])
        self.splits = [DefaultSplit()] + splits if splits else []
        self.cupac_config = cupac_config or {}
        self.custom_analysis_type_mapper = custom_analysis_type_mapper or {}

        self.analysis_type_mapper = self.custom_analysis_type_mapper or analysis_mapping
        self.analysis_class = self.analysis_type_mapper[self.analysis_type]
        self.is_cupac = bool(cupac_config)
        self.ml_handler = self._build_ml_handler(cupac_config, ml_option)
        self.cupac_handler = self.ml_handler  # backward-compat alias
        self.cupac_covariate_col = self.ml_handler.cupac_outcome_name

        self.new_analysis_config = None
        self.experiment_analysis = None

    @staticmethod
    def _build_ml_handler(cupac_config: Optional[dict], ml_option: str):
        if not cupac_config:
            return NoOpHandler()
        if ml_option == "mlrate":
            return MLRateHandler(**cupac_config)
        return CupacHandler(**cupac_config)

    def __repr__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import HypothesisTest, SimpleMetric
        m = SimpleMetric(alias="avg_salary", name="salary")
        ht = HypothesisTest(metric=m, analysis_type="ols")
        print(repr(ht))
        ```
        """
        return (
            f"HypothesisTest(metric={self.metric.alias!r}, analysis_type={self.analysis_type!r}, "
            f"dimensions={len(self.dimensions)}, cupac={self.is_cupac})"
        )

    def __str__(self) -> str:
        """
        Usage:
        ```python
        from cluster_experiments import HypothesisTest, SimpleMetric
        m = SimpleMetric(alias="avg_salary", name="salary")
        ht = HypothesisTest(metric=m, analysis_type="ols")
        print(ht)
        ```
        """
        return f"HypothesisTest(metric={self.metric.alias}, analysis_type={self.analysis_type})"

    def summary(self) -> str:
        """Return a summary of the hypothesis test configuration.

        Usage:
        ```python
        from cluster_experiments import HypothesisTest, SimpleMetric
        m = SimpleMetric(alias="avg_salary", name="salary")
        ht = HypothesisTest(metric=m, analysis_type="ols")
        print(ht.summary())
        ```
        """
        dim_info = ", ".join(d.name for d in self.dimensions)
        lines = [
            "Hypothesis test",
            f"  Metric: {self.metric.alias}",
            f"  Analysis type: {self.analysis_type}",
            f"  Dimensions: {dim_info}",
            f"  CUPAC: {self.is_cupac}",
        ]
        if self.analysis_config:
            lines.append(f"  Analysis config: {self.analysis_config}")
        return "\n".join(lines)

    @staticmethod
    def _validate_inputs(
        metric: Metric,
        analysis_type: str,
        analysis_config: Optional[dict],
        dimensions: Optional[List[Dimension]],
        splits: Optional[List[Split]] = None,
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
        splits : Optional[List[Split]]
            An optional list of Split instances
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
                f"Dimensions must be a list of Dimension instances if provided, got {dimensions}"
            )

        # Check if splits is a list of Split instances when provided
        if splits is not None and (
            not isinstance(splits, list)
            or not all(isinstance(split, Split) for split in splits)
        ):
            raise TypeError(
                f"Splits must be a list of Split instances if provided, got {splits}"
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

        # If a RatioMetric is provided, the only analysis_type allowed is 'delta'
        if isinstance(metric, RatioMetric) and analysis_type != "delta":
            raise ValueError("RatioMetric can only be used with analysis_type 'delta'")

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

    def _prepare_analysis_config(self, treatment_col: str, treatment: str) -> None:
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

        new_analysis_config["target_col"] = self.metric.target_column
        new_analysis_config["treatment_col"] = treatment_col
        new_analysis_config["treatment"] = treatment
        if self.metric.scale_column:
            new_analysis_config["scale_col"] = self.metric.scale_column

        covariates = new_analysis_config.get("covariates", [])

        if self.cupac_covariate_col and self.cupac_covariate_col not in covariates:
            raise ValueError(
                f"You provided a cupac configuration but did not provide the cupac covariate called {self.cupac_covariate_col} in the analysis_config"
            )

        self.new_analysis_config = new_analysis_config

    @staticmethod
    def _aggregate_by_cluster(
        df: pd.DataFrame,
        cluster_cols: List[str],
        treatment_col: str,
        metric: Metric,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate metric values by cluster.
        """
        agg_cols = {}
        if isinstance(metric, RatioMetric):
            agg_cols[metric.target_column] = "sum"
            agg_cols[metric.scale_column] = "sum"
        else:
            agg_cols[metric.target_column] = "sum"

        if covariates:
            for covariate in covariates:
                if covariate not in df.columns:
                    raise ValueError(
                        f"Covariate '{covariate}' is not present in the data for cluster aggregation"
                    )
                agg_cols[covariate] = "mean"

        return df.groupby(cluster_cols + [treatment_col], as_index=False).agg(agg_cols)

    @staticmethod
    def prepare_data(
        data: pd.DataFrame,
        variant_col: str,
        treatment_variant: Variant,
        control_variant: Variant,
        dimension_name: str,
        dimension_value: str,
        split_name: Optional[str] = None,
        split_value: Optional[str] = None,
        cluster_cols: Optional[List[str]] = None,
        metric: Optional[Metric] = None,
        covariates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        prepared_df = data.copy()

        prepared_df = prepared_df.assign(__total_dimension="total")
        prepared_df = prepared_df.query(
            f"{variant_col}.isin(['{treatment_variant.name}','{control_variant.name}'])"
        ).query(f"{dimension_name} == '{dimension_value}'")

        if split_name is not None:
            prepared_df = prepared_df.assign(__total_split="total")
            if split_value is None:
                raise ValueError("split_value must be provided when split_name is used")

            prepared_df = prepared_df.query(f"{split_name} == '{split_value}'")

            if not cluster_cols:
                raise ValueError(
                    f"Split '{split_name}' requires 'cluster_cols' for aggregation."
                )

            prepared_df = HypothesisTest._aggregate_by_cluster(
                df=prepared_df,
                cluster_cols=cluster_cols,
                treatment_col=variant_col,
                metric=metric,
                covariates=covariates,
            )

        return prepared_df

    def add_covariates(
        self, exp_data: pd.DataFrame, pre_exp_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds covariates to the experimental data via the configured handler.
        """
        return self.ml_handler.add_covariates(
            df=exp_data, pre_experiment_df=pre_exp_data
        )

    def get_test_results(
        self,
        control_variant: Variant,
        treatment_variant: Variant,
        variant_col: str,
        exp_data: pd.DataFrame,
        dimension: Dimension,
        dimension_value: str,
        alpha: float,
        split: Optional[Split] = None,
        split_value: Optional[str] = None,
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
        split : Optional[Split], optional
            The split instance to use for segmented analysis and cluster aggregation,
            by default None
        split_value : Optional[str], optional
            The specific value of the split to filter on, by default None

        Returns
        -------
        AnalysisPlanResults
            The results of the hypothesis test
        """
        self._prepare_analysis_config(
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
            split_name=split.name if split else None,
            split_value=split_value,
            cluster_cols=self.analysis_config.get("cluster_cols"),
            metric=self.metric,
            covariates=self.analysis_config.get("covariates", []),
        )

        inference_results = self.get_inference_results(df=prepared_df, alpha=alpha)

        control_variant_mean = self.metric.get_mean(
            prepared_df.query(f"{variant_col}=='{control_variant.name}'")
        )
        treatment_variant_mean = self.metric.get_mean(
            prepared_df.query(f"{variant_col}=='{treatment_variant.name}'")
        )

        has_real_dimensions = any(
            not isinstance(d, DefaultDimension) for d in self.dimensions
        )
        has_real_splits = any(not isinstance(s, DefaultSplit) for s in self.splits)

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
            dimension_name=(
                [dimension.name] if has_real_dimensions else ["__total_dimension"]
            ),
            dimension_value=[dimension_value] if has_real_dimensions else ["total"],
            split_name=(
                [split.name if split else "total"]
                if has_real_splits
                else ["__total_split"]
            ),
            split_value=(
                [split_value if split_value else "total"]
                if has_real_splits
                else ["total"]
            ),
            alpha=[alpha],
        )

        return test_results

    @classmethod
    def from_config(cls, config: dict) -> "HypothesisTest":
        """
        Class method to create an HypothesisTest instance from a configuration dictionary.

        Parameters
        ----------
        config : dict
            A dictionary containing the configuration of the HypothesisTest

        Returns
        -------
        Metric
            A HypothesisTest instance
        """
        metric = Metric.from_metrics_config(config["metric"])
        dimensions = [
            Dimension.from_metrics_config(dimension_config)
            for dimension_config in config.get("dimensions", [])
        ]
        splits = [
            Split.from_metrics_config(split_config)
            for split_config in config.get("splits", [])
        ]
        return cls(
            metric=metric,
            analysis_type=config["analysis_type"],
            analysis_config=config.get("analysis_config"),
            dimensions=dimensions,
            splits=splits,
            cupac_config=config.get("cupac_config"),
            ml_option=config.get("ml_option", "cupac"),
            custom_analysis_type_mapper=config.get("custom_analysis_type_mapper"),
        )
