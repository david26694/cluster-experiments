from dataclasses import dataclass
from typing import List


@dataclass
class Dimension:
    """
    A class used to represent a Dimension with a name and values.

    Attributes
    ----------
    name : str
        The name of the dimension
    values : List[str]
        A list of strings representing the possible values of the dimension

    Methods
    -------
    __post_init__(self):
        Validates the inputs after initialization.
    _validate_inputs(name: str, values: List[str]):
        Validates the inputs for the Dimension class.
    """

    name: str
    values: List[str]

    def __post_init__(self):
        """
        Validates the inputs after initialization.
        """
        self._validate_inputs(self.name, self.values)

    @staticmethod
    def _validate_inputs(name: str, values: List[str]):
        """
        Validates the inputs for the Dimension class.

        Parameters
        ----------
        name : str
            The name of the dimension
        values : List[str]
            A list of strings representing the possible values of the dimension

        Raises
        ------
        TypeError
            If the name is not a string or if values is not a list of strings.
        """
        if not isinstance(name, str):
            raise TypeError("Dimension name must be a string")
        if not isinstance(values, list) or not all(
            isinstance(val, str) for val in values
        ):
            raise TypeError("Dimension values must be a list of strings")


@dataclass
class DefaultDimension(Dimension):
    """
    A class used to represent a Dimension with a default value representing total, i.e. no slicing.
    """

    def __init__(self):
        super().__init__(name="total_dimension", values=["total"])
