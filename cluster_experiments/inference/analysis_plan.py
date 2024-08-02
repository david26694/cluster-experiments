from typing import Any, Dict, List, Optional

import pandas as pd

from cluster_experiments.inference.analysis_results import (
    AnalysisPlanResults,
    EmptyAnalysisPlanResults,
)
from cluster_experiments.inference.dimension import Dimension
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import Metric
from cluster_experiments.inference.variant import Variant


class AnalysisPlan:
    """
    A class used to represent an Analysis Plan with a list of hypothesis tests and a list of variants.
    All the hypothesis tests in the same analysis plan will be analysed with the same dataframe, which will need to be passed in the analyze() method.

    Attributes
    ----------
    tests : List[HypothesisTest]
        A list of HypothesisTest instances
    variants : List[Variant]
        A list of Variant instances
    variant_col : str
        name of the column with the experiment groups
    alpha : float
        significance level used to construct confidence intervals
    """

    def __init__(
        self,
        tests: List[HypothesisTest],
        variants: List[Variant],
        variant_col: str = "treatment",
        alpha: float = 0.05,
    ):
        """
        Parameters
        ----------
        tests : List[HypothesisTest]
            A list of HypothesisTest instances
        variants : List[Variant]
            A list of Variant instances
        variant_col : str
            The name of the column containing the variant names.
        alpha : float
            significance level used to construct confidence intervals
        """

        self.tests = tests
        self.variants = variants
        self.variant_col = variant_col
        self.alpha = alpha

        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates the inputs for the AnalysisPlan class.

        Raises
        ------
        TypeError
            If tests is not a list of HypothesisTest instances or if variants is not a list of Variant instances.
        ValueError
            If tests or variants are empty lists.
        """
        if not isinstance(self.tests, list) or not all(
            isinstance(test, HypothesisTest) for test in self.tests
        ):
            raise TypeError("Tests must be a list of HypothesisTest instances")
        if not isinstance(self.variants, list) or not all(
            isinstance(variant, Variant) for variant in self.variants
        ):
            raise TypeError("Variants must be a list of Variant instances")
        if not isinstance(self.variant_col, str):
            raise TypeError("Variant_col must be a string")
        if not self.tests:
            raise ValueError("Tests list cannot be empty")
        if not self.variants:
            raise ValueError("Variants list cannot be empty")

    def analyze(
        self, exp_data: pd.DataFrame, pre_exp_data: Optional[pd.DataFrame] = None
    ) -> AnalysisPlanResults:

        self._validate_data(exp_data, pre_exp_data)

        analysis_results = EmptyAnalysisPlanResults()

        for test in self.tests:
            if test.is_cupac:
                exp_data = test.cupac_handler.add_covariates(
                    df=exp_data, pre_experiment_df=pre_exp_data
                )

            target_col = test.metric.get_target_column_from_metric()

            for treatment_variant in self.treatment_variants:
                for dimension in test.dimensions:
                    for dimension_value in dimension.iterate_dimension_values():

                        test._prepare_analysis_config(
                            target_col=target_col,
                            treatment_col=self.variant_col,
                            treatment=treatment_variant.name,
                        )

                        prepared_df = test.prepare_data(
                            data=exp_data,
                            variant_col=self.variant_col,
                            treatment_variant=treatment_variant,
                            control_variant=self.control_variant,
                            dimension_name=dimension.name,
                            dimension_value=dimension_value,
                        )

                        inference_results = test.get_inference_results(
                            df=prepared_df, alpha=self.alpha
                        )

                        control_variant_mean = test.metric.get_mean(
                            prepared_df.query(
                                f"{self.variant_col}=='{self.control_variant.name}'"
                            )
                        )
                        treatment_variant_mean = test.metric.get_mean(
                            prepared_df.query(
                                f"{self.variant_col}=='{treatment_variant.name}'"
                            )
                        )

                        test_results = AnalysisPlanResults(
                            metric_alias=[test.metric.alias],
                            control_variant_name=[self.control_variant.name],
                            treatment_variant_name=[treatment_variant.name],
                            control_variant_mean=[control_variant_mean],
                            treatment_variant_mean=[treatment_variant_mean],
                            analysis_type=[test.analysis_type],
                            ate=[inference_results.ate],
                            ate_ci_lower=[inference_results.conf_int.lower],
                            ate_ci_upper=[inference_results.conf_int.upper],
                            p_value=[inference_results.p_value],
                            std_error=[inference_results.std_error],
                            dimension_name=[dimension.name],
                            dimension_value=[dimension_value],
                            alpha=[self.alpha],
                        )

                        analysis_results = analysis_results + test_results

        return analysis_results

    def _validate_data(
        self, exp_data: pd.DataFrame, pre_exp_data: Optional[pd.DataFrame] = None
    ):
        """
        Validates the input dataframes for the analyze method.

        Parameters
        ----------
        exp_data : pd.DataFrame
            The experimental data
        pre_exp_data : Optional[pd.DataFrame]
            The pre-experimental data (optional)

        Raises
        ------
        ValueError
            If exp_data is not a DataFrame or is empty
            If pre_exp_data is provided and is not a DataFrame or is empty
        """
        if not isinstance(exp_data, pd.DataFrame):
            raise ValueError("exp_data must be a pandas DataFrame")
        if exp_data.empty:
            raise ValueError("exp_data cannot be empty")
        if pre_exp_data is not None:
            if not isinstance(pre_exp_data, pd.DataFrame):
                raise ValueError("pre_exp_data must be a pandas DataFrame if provided")
            if pre_exp_data.empty:
                raise ValueError("pre_exp_data cannot be empty if provided")

    @property
    def control_variant(self) -> Variant:
        """
        Returns the control variant from the list of variants. Raises an error if no control variant is found.

        Returns
        -------
        Variant
            The control variant

        Raises
        ------
        ValueError
            If no control variant is found
        """
        for variant in self.variants:
            if variant.is_control:
                return variant
        raise ValueError("No control variant found")

    @property
    def treatment_variants(self) -> List[Variant]:
        """
        Returns the treatment variants from the list of variants. Raises an error if no treatment variants are found.

        Returns
        -------
        List[Variant]
            A list of treatment variants

        Raises
        ------
        ValueError
            If no treatment variants are found
        """
        treatments = [variant for variant in self.variants if not variant.is_control]
        if not treatments:
            raise ValueError("No treatment variants found")
        return treatments

    @classmethod
    def from_metrics(
        cls,
        metrics: List[Metric],
        variants: List[Variant],
        variant_col: str = "treatment",
        alpha: float = 0.05,
        dimensions: Optional[List[Dimension]] = None,
        analysis_type: str = "default",
        analysis_config: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisPlan":
        """
        Creates a simplified AnalysisPlan instance from a list of metrics. It will create HypothesisTest objects under the hood.
        This shortcut does not support cupac, and uses the same dimensions, analysis type and analysis config for all metrics.

        Parameters
        ----------
        metrics : List[Metric]
            A list of Metric instances
        variants : List[Variant]
            A list of Variant instances
        variant_col : str
            The name of the column containing the variant names.
        alpha : float
            Significance level used to construct confidence intervals
        dimensions : Optional[List[Dimension]]
            A list of Dimension instances (optional)
        analysis_type : str
            The type of analysis to be conducted (default: "default")
        analysis_config : Optional[Dict[str, Any]]
            A dictionary containing analysis configuration options (optional)

        Returns
        -------
        AnalysisPlan
            An instance of AnalysisPlan
        """
        tests = [
            HypothesisTest(
                metric=metric,
                dimensions=dimensions or [],
                analysis_type=analysis_type,
                analysis_config=analysis_config or {},
            )
            for metric in metrics
        ]

        return cls(
            tests=tests,
            variants=variants,
            variant_col=variant_col,
            alpha=alpha,
        )
