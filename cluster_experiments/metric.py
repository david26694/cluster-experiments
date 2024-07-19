class Metric:
    """
    A class used to represent a Metric with an alias and components.

    Attributes
    ----------
    alias : str
        A string representing the alias of the metric
    components : tuple
        A tuple of strings representing the components of the metric

    Methods
    -------
    __init__(self, alias: str, components: tuple):
        Initializes the Metric with the provided alias and components.
    _validate_inputs(alias: str, components: tuple):
        Validates the inputs for the Metric class.
    """

    def __init__(self, alias: str, components: tuple):
        """
        Parameters
        ----------
        alias : str
            The alias of the metric
        components : tuple
            A tuple of strings representing the components of the metric
        """
        self._validate_inputs(alias, components)
        self.alias = alias
        self.components = components

    @staticmethod
    def _validate_inputs(alias: str, components: tuple):
        """
        Validates the inputs for the Metric class.

        Parameters
        ----------
        alias : str
            The alias of the metric
        components : tuple
            A tuple of strings representing the components of the metric

        Raises
        ------
        TypeError
            If the alias is not a string or if components is not a tuple of strings.
        """
        if not isinstance(alias, str):
            raise TypeError("Metric alias must be a string")
        if not isinstance(components, tuple) or not all(
            isinstance(comp, str) for comp in components
        ):
            raise TypeError("Metric components must be a tuple of strings")
