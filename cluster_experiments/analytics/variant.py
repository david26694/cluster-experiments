class Variant:
    """
    A class used to represent a Variant with a name and a control flag.

    Attributes
    ----------
    name : str
        The name of the variant
    is_control : bool
        A boolean indicating if the variant is a control variant

    Methods
    -------
    __init__(self, name: str, is_control: bool):
        Initializes the Variant with the provided name and control flag.
    _validate_inputs(name: str, is_control: bool):
        Validates the inputs for the Variant class.
    """

    def __init__(self, name: str, is_control: bool):
        """
        Parameters
        ----------
        name : str
            The name of the variant
        is_control : bool
            A boolean indicating if the variant is a control variant
        """
        self._validate_inputs(name, is_control)
        self.name = name
        self.is_control = is_control

    @staticmethod
    def _validate_inputs(name: str, is_control: bool):
        """
        Validates the inputs for the Variant class.

        Parameters
        ----------
        name : str
            The name of the variant
        is_control : bool
            A boolean indicating if the variant is a control variant

        Raises
        ------
        TypeError
            If the name is not a string or if is_control is not a boolean.
        """
        if not isinstance(name, str):
            raise TypeError("Variant name must be a string")
        if not isinstance(is_control, bool):
            raise TypeError("Variant is_control must be a boolean")
