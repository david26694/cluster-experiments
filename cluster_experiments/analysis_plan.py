from typing import List

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
