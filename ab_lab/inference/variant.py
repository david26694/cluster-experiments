from dataclasses import dataclass


@dataclass
class Variant:
    """
    A class used to represent a Variant with a name and a control flag.

    Attributes
    ----------
    name : str
        The name of the variant
    is_control : bool
        A boolean indicating if the variant is a control variant
    """

    name: str
    is_control: bool

    def __post_init__(self):
        """
        Validates the inputs after initialization.
        """
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates the inputs for the Variant class.

        Raises
        ------
        TypeError
            If the name is not a string or if is_control is not a boolean.
        """
        if not isinstance(self.name, str):
            raise TypeError("Variant name must be a string")
        if not isinstance(self.is_control, bool):
            raise TypeError("Variant is_control must be a boolean")
