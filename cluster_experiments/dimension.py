from typing import List


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
    __init__(self, name: str, values: List[str]):
        Initializes the Dimension with the provided name and values.
    _validate_inputs(name: str, values: List[str]):
        Validates the inputs for the Dimension class.
    """

    def __init__(self, name: str, values: List[str]):
        """
        Parameters
        ----------
        name : str
            The name of the dimension
        values : List[str]
            A list of strings representing the possible values of the dimension
        """
        self._validate_inputs(name, values)
        self.name = name
        self.values = values

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
