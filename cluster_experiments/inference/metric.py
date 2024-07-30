from abc import ABC, abstractmethod

import pandas as pd


class Metric(ABC):
    """
    An abstract base class used to represent a Metric with an alias.

    Attributes
    ----------
    alias : str
        A string representing the alias of the metric
    """

    def __init__(self, alias: str):
        """
        Parameters
        ----------
        alias : str
            The alias of the metric
        """
        self.alias = alias
        self._validate_alias()

    def _validate_alias(self):
        """
        Validates the alias input for the Metric class.

        Raises
        ------
        TypeError
            If the alias is not a string
        """
        if not isinstance(self.alias, str):
            raise TypeError("Metric alias must be a string")

    @abstractmethod
    def get_target_column_from_metric(self) -> str:
        """
        Abstract method to return the target column to feed the experiment analysis class, from the metric definition.

        Returns
        -------
        str
            The target column name
        """
        pass

    @abstractmethod
    def get_mean(self, df: pd.DataFrame) -> float:
        """
        Abstract method to return the mean value of the metric, given a dataframe.

        Returns
        -------
        float
            The mean value of the metric
        """
        pass


class SimpleMetric(Metric):
    """
    A class used to represent a Simple Metric with an alias and a name.
    To be used when the metric is defined at the same level of the data used for the analysis.

    Example
    ----------
    In a clustered experiment the participants were randomised based on their country of residence.
    The metric of interest is the salary of each participant. If the dataset fed into the analysis is at participant-level,
    then a SimpleMetric must be used. However, if the dataset fed into the analysis is at country-level, then a RatioMetric must be used.

    Attributes
    ----------
    alias : str
        A string representing the alias of the metric
    name : str
        A string representing the name of the metric
    """

    def __init__(self, alias: str, name: str):
        """
        Parameters
        ----------
        alias : str
            The alias of the metric
        name : str
            The name of the metric
        """
        super().__init__(alias)
        self.name = name
        self._validate_name()

    def _validate_name(self):
        """
        Validates the name input for the SimpleMetric class.

        Raises
        ------
        TypeError
            If the name is not a string
        """
        if not isinstance(self.name, str):
            raise TypeError("SimpleMetric name must be a string")

    def get_target_column_from_metric(self) -> str:
        """
        Returns the target column for the SimpleMetric.

        Returns
        -------
        str
            The name of the metric
        """
        return self.name

    def get_mean(self, df: pd.DataFrame) -> float:
        """
        Returns the mean value of the metric, given a dataframe.

        Returns
        -------
        float
            The mean value of the metric
        """
        return df[self.name].mean()


class RatioMetric(Metric):
    """
    A class used to represent a Ratio Metric with an alias, a numerator name, and a denominator name.
    To be used when the metric is defined at a lower level than the data used for the analysis.

    Example
    ----------
    In a clustered experiment the participants were randomised based on their country of residence.
    The metric of interest is the salary of each participant. If the dataset fed into the analysis is at country-level,
    then a RatioMetric must be used: the numerator would be the sum of all salaries in the country,
    the denominator would be the number of participants in the country.

    Attributes
    ----------
    alias : str
        A string representing the alias of the metric
    numerator_name : str
        A string representing the numerator name of the metric
    denominator_name : str
        A string representing the denominator name of the metric
    """

    def __init__(self, alias: str, numerator_name: str, denominator_name: str):
        """
        Parameters
        ----------
        alias : str
            The alias of the metric
        numerator_name : str
            The numerator name of the metric
        denominator_name : str
            The denominator name of the metric
        """
        super().__init__(alias)
        self.numerator_name = numerator_name
        self.denominator_name = denominator_name
        self._validate_names()

    def _validate_names(self):
        """
        Validates the numerator and denominator names input for the RatioMetric class.

        Raises
        ------
        TypeError
            If the numerator or denominator names are not strings
        """
        if not isinstance(self.numerator_name, str) or not isinstance(
            self.denominator_name, str
        ):
            raise TypeError("RatioMetric names must be strings")

    def get_target_column_from_metric(self) -> str:
        """
        Returns the target column for the RatioMetric.

        Returns
        -------
        str
            The numerator name of the metric
        """
        return self.numerator_name

    def get_mean(self, df: pd.DataFrame) -> float:
        """
        Returns the mean value of the metric, given a dataframe.

        Returns
        -------
        float
            The mean value of the metric
        """
        return df[self.numerator_name].mean() / df[self.denominator_name].mean()
