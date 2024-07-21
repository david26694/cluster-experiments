from typing import List

import pandas as pd
from pandas import DataFrame

from cluster_experiments.analysis_results import (
    AnalysisPlanResults,
    HypothesisTestResults,
)
from cluster_experiments.hypothesis_test import HypothesisTest
from cluster_experiments.power_config import analysis_mapping
from cluster_experiments.variant import Variant


class AnalysisPlan:
    """
    A class used to represent an Analysis Plan with a list of hypothesis tests and a list of variants.

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

    Methods
    -------
    __init__(self, tests: List[HypothesisTest], variants: List[Variant]):
        Initializes the AnalysisPlan with the provided list of hypothesis tests and variants.
    _validate_inputs(tests: List[HypothesisTest], variants: List[Variant]):
        Validates the inputs for the AnalysisPlan class.
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
        self._validate_inputs(tests, variants, variant_col)
        self.tests = tests
        self.variants = variants
        self.variant_col = variant_col
        self.alpha = alpha

    @staticmethod
    def _validate_inputs(
        tests: List[HypothesisTest], variants: List[Variant], variant_col: str
    ):
        """
        Validates the inputs for the AnalysisPlan class.

        Parameters
        ----------
        tests : List[HypothesisTest]
            A list of HypothesisTest instances
        variants : List[Variant]
            A list of Variant instances
        variant_col : str
            The name of the column containing the variant names.

        Raises
        ------
        TypeError
            If tests is not a list of HypothesisTest instances or if variants is not a list of Variant instances.
        ValueError
            If tests or variants are empty lists.
        """
        if not isinstance(tests, list) or not all(
            isinstance(test, HypothesisTest) for test in tests
        ):
            raise TypeError("Tests must be a list of HypothesisTest instances")
        if not isinstance(variants, list) or not all(
            isinstance(variant, Variant) for variant in variants
        ):
            raise TypeError("Variants must be a list of Variant instances")
        if not isinstance(variant_col, str):
            raise TypeError("Variant_col must be a string")
        if not tests:
            raise ValueError("Tests list cannot be empty")
        if not variants:
            raise ValueError("Variants list cannot be empty")

    def analyze(
        self,
        exp_data: pd.DataFrame,  # , pre_exp_data: Optional[pd.DataFrame], alpha=0.05
    ) -> DataFrame:
        ...
        # add methods to prepare the filtered dataset based on variants and slicers
        # add methods to run the analysis for each of the hypothesis tests, given a filtered dataset
        # store each row as a HypothesisTestResults object
        # wrap all results in an AnalysisPlanResults object

        # -----

        # add all kind of checks on the inputs at the beginning using the data structures
        # todo: ...
        # do it before running the computations below

        test_results = []
        treatment_variants: List[Variant] = self.get_treatment_variants()
        control_variant: Variant = self.get_control_variant()

        for test in self.tests:
            # todo: add cupac handler here
            analysis_class = analysis_mapping[test.analysis_type]
            target_col = test.metric.get_target_column_from_metric()

            for treatment_variant in treatment_variants:

                analysis_config_final = self.prepare_analysis_config(
                    initial_analysis_config=test.analysis_config,
                    target_col=target_col,
                    treatment_col=self.variant_col,
                    treatment=treatment_variant.name,
                )

                experiment_analysis = analysis_class(**analysis_config_final)

                for dimension in test.dimensions:
                    for dimension_value in list(set(dimension.values)):
                        prepared_df = self.prepare_data(
                            data=exp_data,
                            variant_col=self.variant_col,
                            treatment_variant=treatment_variant,
                            control_variant=control_variant,
                            dimension_name=dimension.name,
                            dimension_value=dimension_value,
                        )

                        ate = experiment_analysis.get_point_estimate(df=prepared_df)
                        p_value = experiment_analysis.get_pvalue(df=prepared_df)
                        std_error = experiment_analysis.get_standard_error(
                            df=prepared_df
                        )
                        confidence_interval = (
                            experiment_analysis.get_confidence_interval(
                                df=prepared_df, alpha=self.alpha
                            )
                        )
                        control_variant_mean = test.metric.get_mean(
                            prepared_df.query(
                                f"{self.variant_col}=='{control_variant.name}'"
                            )
                        )
                        treatment_variant_mean = test.metric.get_mean(
                            prepared_df.query(
                                f"{self.variant_col}=='{treatment_variant.name}'"
                            )
                        )

                        test_results.append(
                            HypothesisTestResults(
                                metric_alias=test.metric.alias,
                                control_variant_name=control_variant.name,
                                treatment_variant_name=treatment_variant.name,
                                control_variant_mean=control_variant_mean,
                                treatment_variant_mean=treatment_variant_mean,
                                analysis_type=test.analysis_type,
                                ate=ate,
                                ate_ci_lower=confidence_interval.lower,
                                ate_ci_upper=confidence_interval.upper,
                                p_value=p_value,
                                std_error=std_error,
                                dimension_name=dimension.name,
                                dimension_value=dimension_value,
                                alpha=self.alpha,
                            )
                        )

        return AnalysisPlanResults.from_results(test_results)

    def prepare_data(
        self,
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

        prepared_df = prepared_df.query(
            f"{variant_col}.isin(['{treatment_variant.name}','{control_variant.name}'])"
        ).query(f"{dimension_name} == '{dimension_value}'")

        return prepared_df

    def get_control_variant(self) -> Variant:
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

    def get_treatment_variants(self) -> List[Variant]:
        """
        Returns the treatment variants from the list of variants.

        Returns
        -------
        List[Variant]
            A list of treatment variants
        """
        return [variant for variant in self.variants if not variant.is_control]

    @staticmethod
    def prepare_analysis_config(
        initial_analysis_config: dict,
        target_col: str,
        treatment_col: str,
        treatment: str,
    ) -> dict:
        """
        Extends the analysis_config provided by the user, by adding or overriding the following keys:
        - target_col
        - treatment_col
        - treatment

        Returns
        -------
        dict
            The prepared analysis configuration, ready to be ingested by the experiment analysis class
        """
        initial_analysis_config["target_col"] = target_col
        initial_analysis_config["treatment_col"] = treatment_col
        initial_analysis_config["treatment"] = treatment

        return initial_analysis_config
