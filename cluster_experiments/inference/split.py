from dataclasses import dataclass

from cluster_experiments.inference.dimension import Dimension


@dataclass
class Split(Dimension):
    """
    A class used to represent a Split with a name and values.

    Splits describe attributes that may change during the course of the experiment.
    """

    @classmethod
    def from_metrics_config(cls, config: dict) -> "Split":
        return cls(name=config["name"], values=config["values"])


@dataclass
class DefaultSplit(Dimension):
    """
    A class used to represent a Split with a default value representing total, i.e. no slicing.

    DefaultSplit is used when no explicit split grouping is requested and the analysis
    should consider the total population or aggregated cluster values.
    """

    def __init__(self):
        super().__init__(name="__total_split", values=["total"])

    def __str__(self) -> str:
        return "DefaultSplit(total)"
