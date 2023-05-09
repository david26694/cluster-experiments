from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
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
        target_col: name of the target_col to use as the outcome
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


class StochasticPerturbator(Perturbator):
    """The StochasticPerturbator class implements a perturbator that adds a stochastic effect
    to the target column of the treated instances. The stochastic effect is sampled from a
    normal distribution with mean average_effect and variance scale. If scale is not
    provided, the variance is abs(average_effect).

    Arguments:
        average_effect (Optional[float], optional): the average effect of the treatment. Defaults to None.
        target_col (str, optional): name of the target_col to use as the outcome. Defaults to "target".
        treatment_col (str, optional): the name of the column that contains the treatment. Defaults to "treatment".
        treatment (str, optional): name of the treatment to use as the treated group. Defaults to "B".
        scale (Optional[float], optional): the scale of the effect distribution. Defaults to None.
    """

    def __init__(
        self,
        average_effect: Optional[float] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        scale: Optional[float] = None,
    ):
        super().__init__(average_effect, target_col, treatment_col, treatment)
        self._scale = scale

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """Perturbate with a normal effect with mean average_effect and
        std abs(average_effect).

        Arguments:
            df (pd.DataFrame): the dataframe to perturbate.
            average_effect (Optional[float], optional): the average effect. Defaults to None.

        Returns:
            pd.DataFrame: the perturbated dataframe.

        Usage:

        ```python
        from cluster_experiments.perturbator import StochasticPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = NormalPerturbator()
        perturbator.perturbate(df, average_effect=1)
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        scale = self._get_scale(average_effect)
        n = self._get_number_of_treated(df)
        sampled_effect = self._sample_normal_effect(average_effect, scale, n)
        df = self._apply_additive_effect(df, sampled_effect)
        return df

    def _get_scale(self, average_effect: float) -> float:
        """Get the scale of the normal distribution. If scale is not provided, the
        variance is abs(average_effect). Raises a ValueError if scale is not positive.
        """
        scale = abs(average_effect) if self._scale is None else self._scale
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        return scale

    def _get_number_of_treated(self, df: pd.DataFrame) -> int:
        """Get the number of treated instances in the dataframe"""
        return (df[self.treatment_col] == self.treatment).sum()

    def _sample_normal_effect(
        self, average_effect: float, scale: float, n: int
    ) -> float:
        return np.random.normal(average_effect, scale, n)

    def _apply_additive_effect(
        self, df: pd.DataFrame, sampled_effect: np.ndarray
    ) -> pd.DataFrame:
        df.loc[
            df[self.treatment_col] == self.treatment, self.target_col
        ] += sampled_effect
        return df

    @classmethod
    def from_config(cls, config):
        """Creates a Perturbator object from a PowerConfig object"""
        return cls(
            average_effect=config.average_effect,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            scale=config.scale,
        )


class RelativePositivePerturbator(Perturbator):
    """
    A Perturbator for continuous, positively-defined targets
    applies a simulated effect multiplicatively for the treated samples, ie.
    proportional to the target value for each sample. The number of samples with 0
    as target remains unchanged.
    """

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:
        ```python
        from cluster_experiments.perturbator import RelativePositivePerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = RelativePositivePerturbator()
        # Increase target metric by 50%
        perturbator.perturbate(df, average_effect=0.5)
        # returns pd.DataFrame({"target": [1, 3, 3], "treatment": ["A", "B", "A"]})
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        self._assert_multiplicative_effect(df, average_effect)
        df = self._apply_multiplicative_effect(df, average_effect)
        return df

    def _assert_multiplicative_effect(
        self, df: pd.DataFrame, average_effect: float
    ) -> None:
        if average_effect < -1.0:
            raise ValueError(
                "Simulated effect needs to be bigger than -100%, got "
                f"{average_effect*100:.1f}%"
            )

        treatment_zeros = (
            (df[self.treatment_col] != self.treatment) | (df[self.target_col] == 0)
        ).mean()
        if 1.0 == treatment_zeros:
            raise ValueError(
                f"All treatment samples have {self.target_col} = 0, relative effect "
                f"{average_effect} will have no effect"
            )

    def _apply_multiplicative_effect(
        self, df: pd.DataFrame, effect: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df.loc[df[self.treatment_col] == self.treatment, self.target_col] *= 1 + effect
        return df


class StochasticRelativePositivePerturbator(
    StochasticPerturbator, RelativePositivePerturbator
):
    """
    A Perturbator for continuous, positively-defined targets that applies a simulated
    stochastic effect multiplicatively for the treated samples, ie. proportional to the
    target value for each sample. The number of samples with 0 as target remains unchanged.

    The stochastic effect is sampled from a beta distribution with parameters mean and
    variance. If variance is not provided, the variance is abs(mean).

    The common beta parameters are derived from the mean and scale as follows:

    ```
    a <- mu / (scale * scale)
    b <- (1-mu) / (scale * scale)
    effect ~ beta(a, b)
    ```
    source: https://stackoverflow.com/a/51143208

    Example: a mean = 0.2 and variance = 0.1, give a = 20 and b = 80
    Plot: https://www.wolframalpha.com/input?i=plot+distribution+of+beta%2820%2C+80%29

    Arguments:
        average_effect (Optional[float], optional): the average effect of the treatment. Defaults to None.
        target_col (str, optional): name of the target_col to use as the outcome. Defaults to "target".
        treatment_col (str, optional): the name of the column that contains the treatment. Defaults to "treatment".
        treatment (str, optional): name of the treatment to use as the treated group. Defaults to "B".
        scale (Optional[float], optional): the scale of the effect distribution. Defaults to None.
    """

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:
        ```python
        from cluster_experiments.perturbator import RelativePositivePerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = StochasticRelativePositivePerturbator()
        # Increase target metric by 50%
        perturbator.perturbate(df, average_effect=0.2)
        # returns pd.DataFrame({"target": [1, 3, 3], "treatment": ["A", "B", "A"]})
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        self._assert_multiplicative_effect(df, average_effect)
        scale = self._get_scale(average_effect)
        n = self._get_number_of_treated(df)
        sampled_effect = self._sample_beta_effect(average_effect, scale, n)
        df = self._apply_multiplicative_effect(df, sampled_effect)
        return df

    def _sample_beta_effect(
        self, average_effect: float, scale: float, n: int
    ) -> np.ndarray:
        a = average_effect / (scale * scale)
        b = (1 - average_effect) / (scale * scale)
        return np.random.beta(a, b, n)


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

    def _data_checks(self, df: pd.DataFrame, average_effect: float) -> None:
        """Check that outcome is indeed binary, and average effect is in (-1, 1)"""

        if set(df[self.target_col].unique()) - {0, 1}:
            raise ValueError(
                f"Target column must be binary, found {set(df[self.target_col].unique())}"
            )

        if average_effect > 1 or average_effect < -1:
            raise ValueError(
                f"Average effect must be in (-1, 1), found {average_effect}"
            )

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

        self._data_checks(df, average_effect)

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
