from typing import List

import pandas as pd

from cluster_experiments.analysis_results import AnalysisPlanResults
from cluster_experiments.hypothesis_test import HypothesisTest
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

    Methods
    -------
    __init__(self, tests: List[HypothesisTest], variants: List[Variant]):
        Initializes the AnalysisPlan with the provided list of hypothesis tests and variants.
    _validate_inputs(tests: List[HypothesisTest], variants: List[Variant]):
        Validates the inputs for the AnalysisPlan class.
    """

    def __init__(self, tests: List[HypothesisTest], variants: List[Variant]):
        """
        Parameters
        ----------
        tests : List[HypothesisTest]
            A list of HypothesisTest instances
        variants : List[Variant]
            A list of Variant instances
        """
        self._validate_inputs(tests, variants)
        self.tests = tests
        self.variants = variants

    @staticmethod
    def _validate_inputs(tests: List[HypothesisTest], variants: List[Variant]):
        """
        Validates the inputs for the AnalysisPlan class.

        Parameters
        ----------
        tests : List[HypothesisTest]
            A list of HypothesisTest instances
        variants : List[Variant]
            A list of Variant instances

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
        if not tests:
            raise ValueError("Tests list cannot be empty")
        if not variants:
            raise ValueError("Variants list cannot be empty")

    def analyze(
        self, exp_data: pd.DataFrame, pre_exp_data: pd.DataFrame, alpha=0.05
    ) -> AnalysisPlanResults:
        ...
        # add methods to prepare the filtered dataset based on variants and slicers
        # add methods to run the analysis for each of the hypothesis tests, given a filtered dataset
        # store each row as a HypothesisTestResults object
        # wrap all results in an AnalysisPlanResults object
