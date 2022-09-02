from abc import ABC, abstractmethod

import pandas as pd


class Perturbator(ABC):
    """Abstract perturbator. Perturbators are used to simulate a fictitious effect when running a power analysis.

    The idea is that, when running a power analysis, we split our instances according to a RandomSplitter, and the
    instances that got the treatment, are perturbated with a fictional effect via the Perturbator.
    """

    def __init__(
        self,
        average_effect: float,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
    ):
        """
        Arguments:
            average_effect: The average effect of the treatment
            treatment: name of the treatment to use as the treated group
            treatment_col: The name of the column that contains the treatment
            treatment: name of the treatment to use as the treated group
        """
        self.average_effect = average_effect
        self.target_col = target_col
        self.treatment_col = treatment_col
        self.treatment = treatment
        self.treated_query = f"{self.treatment_col} == '{self.treatment}'"

    @abstractmethod
    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
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
    """UniformPerturbator is a Perturbator that adds a uniform effect to the target column of the treated instances."""

    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Usage:

        ```python
        from cluster_experiments.perturbator import UniformPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = UniformPerturbator(average_effect=1)
        perturbator.perturbate(df)
        ```
        """
        df = df.copy().reset_index(drop=True)
        df.loc[
            df[self.treatment_col] == self.treatment, self.target_col
        ] += self.average_effect
        return df


class BinaryPerturbator(Perturbator):
    """BinaryPerturbator is a Perturbator that adds is used to deal with binary outcome variables.
    It randomly selects some treated instances and flips their outcome from 0 to 1 or 1 to 0, depending on the effect being positive or negative"""

    def _sample_max(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """Like sample without replacement,
        but if you are to sample more than 100% of the data,
        it just returns the whole dataframe."""
        if n >= len(df):
            return df
        return df.sample(n=n)

    def perturbate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Usage:

        ```python
        from cluster_experiments.perturbator import BinaryPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 0, 1], "treatment": ["A", "B", "A"]})
        perturbator = BinaryPerturbator(average_effect=0.1)
        perturbator.perturbate(df)
        ```
        """

        df = df.copy().reset_index(drop=True)
        from_target, to_target = 1, 0
        if self.average_effect > 0:
            from_target, to_target = 0, 1

        n_transformed = abs(
            int(self.average_effect * len(df.query(self.treated_query)))
        )
        idx = list(
            # Sample of negative cases in group B
            df.query(f"{self.target_col} == {from_target} & {self.treated_query}")
            .pipe(self._sample_max, n=n_transformed)
            .index.drop_duplicates()
        )
        df.loc[idx, self.target_col] = to_target
        return df
