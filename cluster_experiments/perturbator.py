from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Perturbator(ABC):
    """
    Abstract perturbator. Perturbators are used to simulate a fictitious effect when running a power analysis.

    The idea is that, when running a power analysis, we split our instances according to a RandomSplitter, and the
    instances that got the treatment, are perturbated with a fictional effect via the Perturbator.

    In order to create your own perturbator, you should create a derived class that implements the perturbate method.
    The perturbate method should add the average effect in the desired way and return the dataframe with the extra average effect,
    without affecting the initial dataframe. Keep in mind to use `df = df.copy()` in the first line of the perturbate method.

    Arguments:
        average_effect: The average effect of the treatment
        treatment: name of the treatment to use as the treated group
        treatment_col: The name of the column that contains the treatment
        treatment: name of the treatment to use as the treated group

    """

    def __init__(
        self,
        average_effect: Optional[float] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
    ):
        self.average_effect = average_effect
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.treatment = treatment
        self.treated_query = f"{self.treatment_col} == '{self.treatment}'"

    def get_average_effect(self, average_effect: Optional[float] = None) -> float:
        average_effect = (
            average_effect if average_effect is not None else self.average_effect
        )
        if average_effect is None:
            raise ValueError(
                "average_effect must be provided, either in the constructor or in the method call"
            )
        return average_effect

    @abstractmethod
    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """Method to perturbate a dataframe"""
        pass

    @classmethod
    def from_config(cls, config):
        """Creates a Perturbator object from a PowerConfig object"""
        return cls(
            average_effect=config.average_effect,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
        )


class UniformPerturbator(Perturbator):
    """
    UniformPerturbator is a Perturbator that adds a uniform effect to the target column of the treated instances.
    """

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:

        ```python
        from cluster_experiments.perturbator import UniformPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = UniformPerturbator()
        perturbator.perturbate(df, average_effect=1)
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        df.loc[
            df[self.treatment_col] == self.treatment, self.target_col
        ] += average_effect
        return df


class BinaryPerturbator(Perturbator):
    """
    BinaryPerturbator is a Perturbator that adds is used to deal with binary outcome variables.
    It randomly selects some treated instances and flips their outcome from 0 to 1 or 1 to 0, depending on the effect being positive or negative
    """

    def _sample_max(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Like sample without replacement,
        but if you are to sample more than 100% of the data,
        it just returns the whole dataframe."""
        if n >= len(df):
            return df
        return df.sample(n=n)

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:

        ```python
        from cluster_experiments.perturbator import BinaryPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 0, 1], "treatment": ["A", "B", "A"]})
        perturbator = BinaryPerturbator()
        perturbator.perturbate(df, average_effect=0.1)
        ```
        """

        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)

        from_target, to_target = 1, 0
        if average_effect > 0:
            from_target, to_target = 0, 1

        n_transformed = abs(int(average_effect * len(df.query(self.treated_query))))
        idx = list(
            # Sample of negative cases in group B
            df.query(f"{self.target_col} == {from_target} & {self.treated_query}")
            .pipe(self._sample_max, n=n_transformed)
            .index.drop_duplicates()
        )
        df.loc[idx, self.target_col] = to_target
        return df
