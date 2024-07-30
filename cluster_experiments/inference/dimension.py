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
    """

    name: str
    values: List[str]

    def __post_init__(self):
        """
        Validates the inputs after initialization.
        """
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates the inputs for the Dimension class.

        Raises
        ------
        TypeError
            If the name is not a string or if values is not a list of strings.
        """
        if not isinstance(self.name, str):
            raise TypeError("Dimension name must be a string")
        if not isinstance(self.values, list) or not all(
            isinstance(val, str) for val in self.values
        ):
            raise TypeError("Dimension values must be a list of strings")

    def iterate_dimension_values(self):
        """
        A generator method to yield name and values from the dimension.

        Yields
        ------
        Any
            A unique value from the dimension.
        """
        seen = set()
        for value in self.values:
            if value not in seen:
                seen.add(value)
                yield value


@dataclass
class DefaultDimension(Dimension):
    """
    A class used to represent a Dimension with a default value representing total, i.e. no slicing.
    """

    def __init__(self):
        super().__init__(name="__total_dimension", values=["total"])
