import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union

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


class ConstantPerturbator(Perturbator):
    """
    ConstantPerturbator is a Perturbator that adds a constant effect to the target column of the treated instances.
    """

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:

        ```python
        from cluster_experiments.perturbator import ConstantPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = ConstantPerturbator()
        perturbator.perturbate(df, average_effect=1)
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        df = self.apply_additive_effect(df, average_effect)
        return df

    def apply_additive_effect(
        self, df: pd.DataFrame, effect: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df.loc[df[self.treatment_col] == self.treatment, self.target_col] += effect
        return df


class UniformPerturbator(Perturbator):
    """
    UniformPerturbator is a Perturbator that adds a constant effect to the target column of the treated instances.
    """

    def __init__(
        self,
        average_effect: Optional[float] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
    ):
        super().__init__(average_effect, target_col, treatment_col, treatment)
        logging.warning(
            "UniformPerturbator is deprecated and will be removed in future versions. "
            "Use ConstantPerturbator instead."
        )

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
        df = self.apply_additive_effect(df, average_effect)
        return df

    def apply_additive_effect(
        self, df: pd.DataFrame, effect: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df.loc[df[self.treatment_col] == self.treatment, self.target_col] += effect
        return df


class NormalPerturbator(ConstantPerturbator):
    """The NormalPerturbator class implements a perturbator that adds a stochastic effect
    to the target column of the treated instances. The stochastic effect is sampled from a
    normal distribution with mean average_effect and variance scale. If scale is not
    provided, the variance is abs(average_effect). If scale is provided, a
    value not much bigger than the average_effect is suggested.

    ```
    target -> target + effect, where effect ~ Normal(average_effect, scale)
    ```

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
        from cluster_experiments.perturbator import NormalPerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = NormalPerturbator()
        perturbator.perturbate(df, average_effect=1)
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        scale = self.get_scale(average_effect)
        n = self.get_number_of_treated(df)
        sampled_effect = self._sample_normal_effect(average_effect, scale, n)
        df = self.apply_additive_effect(df, sampled_effect)
        return df

    def get_scale(self, average_effect: float) -> float:
        """Get the scale of the normal distribution. If scale is not provided, the
        variance is abs(average_effect). Raises a ValueError if scale is not positive.
        """
        scale = abs(average_effect) if self._scale is None else self._scale
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        return scale

    def get_number_of_treated(self, df: pd.DataFrame) -> int:
        """Get the number of treated instances in the dataframe"""
        return (df[self.treatment_col] == self.treatment).sum()

    def _sample_normal_effect(
        self, average_effect: float, scale: float, n: int
    ) -> np.ndarray:
        return np.random.normal(average_effect, scale, n)

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

    ```
    target -> target * (1 + average_effect), where -1 < average_effect < inf
                                               and target > 0 for all samples
    ```
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
        self.check_relative_positive_effect(df, average_effect)
        df = self.apply_multiplicative_effect(df, average_effect)
        return df

    def check_relative_positive_effect(
        self, df: pd.DataFrame, average_effect: float
    ) -> None:
        self.check_average_effect_greater_than(average_effect, x=-1)
        self.check_target_is_not_negative(df)
        self.check_target_is_not_constant_zero(df, average_effect)

    def check_target_is_not_constant_zero(
        self, df: pd.DataFrame, average_effect: float
    ) -> Optional[NoReturn]:
        treatment_zeros = (
            (df[self.treatment_col] != self.treatment) | (df[self.target_col] == 0)
        ).mean()
        if 1.0 == treatment_zeros:
            raise ValueError(
                f"All treatment samples have {self.target_col} = 0, relative effect "
                f"{average_effect} will have no effect"
            )

    def check_target_is_not_negative(self, df: pd.DataFrame) -> Optional[NoReturn]:
        if any(df[self.target_col] < 0):
            raise ValueError(
                f"All {self.target_col} values need to be positive or 0, "
                f"got {df[self.target_col].min()}"
            )

    def check_average_effect_greater_than(
        self, average_effect: float, x: float
    ) -> Optional[NoReturn]:
        if average_effect <= x:
            raise ValueError(
                f"Simulated effect needs to be greater than {x*100:.0f}%, got "
                f"{average_effect*100:.1f}%"
            )

    def apply_multiplicative_effect(
        self, df: pd.DataFrame, effect: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        df.loc[df[self.treatment_col] == self.treatment, self.target_col] *= 1 + effect
        return df


class BetaRelativePositivePerturbator(NormalPerturbator, RelativePositivePerturbator):
    """
    A stochastic Perturbator for continuous, positively-defined targets that applies a
    sampled effect from the Beta distribution. It applies the effect multiplicatively.

    *WARNING*: the average effect is only defined for values between 0 and 1 (not
    included). Therefore, it only increments the target for the treated samples.

    The number of samples with 0 as target remains unchanged.

    The stochastic effect is sampled from a beta distribution with parameters mean and
    variance. If variance is not provided, the variance is abs(mean). Hence, the effect
    is bounded by 0 and 1.

    ```
    target -> target * (1 + effect), where effect ~ Beta(a, b); a, b > 0
                                       and target > 0 for all samples
    ```

    The common beta parameters are derived from the mean and scale parameters (see
    how below). That's why the average effect is only defined for values between 0
    and 1, otherwise one of the beta parameters would be negative or zero:

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
        from cluster_experiments.perturbator import BetaRelativePositivePerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = BetaRelativePositivePerturbator()
        # Increase target metric by 20%
        perturbator.perturbate(df, average_effect=0.2)
        # returns pd.DataFrame({"target": [1, 3, 3], "treatment": ["A", "B", "A"]})
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        self.check_beta_positive_effect(df, average_effect)
        scale = self.get_scale(average_effect)
        n = self.get_number_of_treated(df)
        sampled_effect = self._sample_beta_effect(average_effect, scale, n)
        df = self.apply_multiplicative_effect(df, sampled_effect)
        return df

    def check_beta_positive_effect(self, df, average_effect):
        self.check_average_effect_greater_than(average_effect, x=0)
        self.check_average_effect_less_than(average_effect, x=1)
        self.check_target_is_not_negative(df)
        self.check_target_is_not_constant_zero(df, average_effect)

    def check_average_effect_less_than(
        self, average_effect: float, x: float
    ) -> Optional[NoReturn]:
        if average_effect >= x:
            raise ValueError(
                f"Simulated effect needs to be less than {x*100:.0f}%, got "
                f"{average_effect*100:.1f}%"
            )

    def _sample_beta_effect(
        self, average_effect: float, scale: float, n: int
    ) -> np.ndarray:
        a = average_effect / (scale * scale)
        b = (1 - average_effect) / (scale * scale)
        return np.random.beta(a, b, n)


class BetaRelativePerturbator(NormalPerturbator, RelativePositivePerturbator):
    """
    A stochastic Perturbator for continuous targets that applies a  sampled
    effect from a scaled Beta distribution. It applies the effect multiplicatively.

    The sampled effect is defined for values in the specified range
    (range_min, range_max). It's recommended to set -1<range_min<0 and
    range_max>0 in a "symmetric way" around 0, such that
    log(1 + range_min) = -log(1 + range_max).
    This ensures to have an "symmetric range" of perturbations that relatively
    decrease the target as perturbations that relatively increase the target.
    By "symmetry" of relative effects we mean that for an effect c > 0, an
    increase of the target t via t*(1 + c) is "symmetric" to a decrease of t
    via t/(1 + c). For example, an increase of 5x (i.e. by +400%, corresponding
    to c_inc=4) is "symmetric" to a decrease of 5x (i.e. a decrease of -80%,
    corresponding to c_dec = -0.8). In this case, 1 + c_dec = 1/(1 + c_inc), so
    the relative effects c_inc and c_dec are "symmetric" in the sense that they
    are inverse to each other.

    The number of samples with 0 as target remains unchanged.

    The stochastic effect is sampled from a beta distribution with parameters
    mean and variance, which is linearly scaled to the range
    (range_min, range_max).
    If variance is not provided, the variance is abs(mean).

    ```
    target -> target * (1 + effect), where effect ~ Beta(a, b)
    ```

    The common beta parameters are derived from the mean and scale parameters,
    combined with linear transformations to ensure the support in the given
    range. The resulting beta parameters are scaled by abs(mu) to narrow the
    beta distribution around the mean.

    ```
    mu_transformed <- (mu - range_min) / (range_max - range_min)
    scale_transformed <- (scale - range_min) / (range_max - range_min)
    a <- mu_transformed / (scale_transformed * scale_transformed)
    b <- (1-mu_transformed) / (scale_transformed * scale_transformed)
    effect_transformed ~ beta(a/abs(mu), b/abs(mu))
    effect = effect_transformed * (range_max - range_min) + range_min
    ```

    Arguments:
        average_effect (Optional[float], optional): the average effect of the treatment. Defaults to None.
        target_col (str, optional): name of the target_col to use as the outcome. Defaults to "target".
        treatment_col (str, optional): the name of the column that contains the treatment. Defaults to "treatment".
        treatment (str, optional): name of the treatment to use as the treated group. Defaults to "B".
        scale (Optional[float], optional): the scale of the effect distribution. Defaults to None.
            If not provided, the variance of the beta distribution is abs(mean).
        range_min (float, optional): the minimum value of the target range, must be >-1.
            Defaults to -0.8, which allows for up to 5x decreases of the target.
        range_max (float, optional): the maximum value of the target range.
            Defaults to 4, which allows for up to 5x increases of the target.
        reduce_variance (Optional[bool], optional): if True and if abs(average_effect)<1, we reduce
            the variance of the beta distribution by multiplying the beta parameters by 1/abs(average_effect).
            Defaults to None, which is equivalent to True.
    """

    def __init__(
        self,
        average_effect: Optional[float] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        scale: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        reduce_variance: Optional[bool] = None,
    ):
        self._check_range(range_min, range_max)
        super().__init__(average_effect, target_col, treatment_col, treatment, scale)
        self._range_min = range_min or -0.8
        self._range_max = range_max or 4
        self._reduce_variance = reduce_variance or True

    def perturbate(
        self, df: pd.DataFrame, average_effect: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Usage:
        ```python
        from cluster_experiments.perturbator import BetaRelativePerturbator
        import pandas as pd
        df = pd.DataFrame({"target": [1, 2, 3], "treatment": ["A", "B", "A"]})
        perturbator = BetaRelativePerturbator(range_min = -0.5, range_max = 2)
        # Increase target metric by 20% on average
        perturbator.perturbate(df, average_effect=0.2)
        ```
        """
        df = df.copy().reset_index(drop=True)
        average_effect = self.get_average_effect(average_effect)
        self.check_relative_effect_bounds(average_effect)
        scale = self.get_scale(average_effect)
        self.check_relative_effect_bounds(scale)
        n = self.get_number_of_treated(df)
        sampled_effect = self._sample_scaled_beta_effect(average_effect, scale, n)
        df = self.apply_multiplicative_effect(df, sampled_effect)
        return df

    @staticmethod
    def _check_range(range_min: float, range_max: float):
        if range_min < -1:
            raise ValueError(f"range_min needs to be greater than -1, got {range_min}")
        if range_min >= range_max:
            raise ValueError(
                f"range_min needs to be smaller than range_max, got "
                f"{range_min = } and {range_max = }"
            )

    def check_relative_effect_bounds(self, average_effect: float) -> None:
        self.check_average_effect_greater_than(average_effect, x=self._range_min)
        self.check_average_effect_smaller_than(average_effect, x=self._range_max)

    def check_average_effect_greater_than(
        self, average_effect: float, x: float
    ) -> Optional[NoReturn]:
        if average_effect <= x:
            raise ValueError(
                f"Simulated effect needs to be greater than range_min={x}, got {average_effect}"
            )

    def check_average_effect_smaller_than(
        self, average_effect: float, x: float
    ) -> Optional[NoReturn]:
        if average_effect >= x:
            raise ValueError(
                f"Simulated effect needs to be smaller than range_max={x}, got {average_effect}"
            )

    def _reduce_variance_beta_params(
        self, average_effect: float, a: float, b: float
    ) -> Tuple[float, float]:
        """
        Multiplying the parameters of the beta distribution with a factor >1
        reduces variance
        """
        if abs(average_effect) < 1:
            a *= 1 / abs(average_effect)
            b *= 1 / abs(average_effect)
        return a, b

    def _sample_scaled_beta_effect(
        self, average_effect: float, scale: float, n: int
    ) -> np.ndarray:
        average_effect_inv_transf = self._inv_transform_to_range(average_effect)
        scale_inv_transf = self._inv_transform_to_range(scale)
        a = average_effect_inv_transf / (scale_inv_transf * scale_inv_transf)
        b = (1 - average_effect_inv_transf) / (scale_inv_transf * scale_inv_transf)

        if self._reduce_variance:
            a, b = self._reduce_variance_beta_params(average_effect, a, b)
        beta = np.random.beta(a, b, n)

        return self._transform_to_range(beta)

    def _transform_to_range(self, x: Union[float, np.ndarray]):
        return x * (self._range_max - self._range_min) + self._range_min

    def _inv_transform_to_range(self, x: Union[float, np.ndarray]):
        return (x - self._range_min) / (self._range_max - self._range_min)

    @classmethod
    def from_config(cls, config):
        """Creates a Perturbator object from a PowerConfig object"""
        return cls(
            average_effect=config.average_effect,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            scale=config.scale,
            range_min=config.range_min,
            range_max=config.range_max,
        )


class SegmentedBetaRelativePerturbator(BetaRelativePositivePerturbator):
    """
    A stochastic Perturbator for continuous targets that applies a sampled
    effect from the Beta distribution. It applies the effect multiplicatively
    and based on given segments.
    For each segment, the average segment effect is sampled from a beta
    distribution with support in (0, 1). Within each segment, the individual
    effects are sampled from a beta distribution with mean equal to the segment
    average effect and support in (range_min, range_max).

    The number of samples with 0 as target remains unchanged.

    For additional details and recommendations on the parameters, see the
    documentation for the `BetaRelativePerturbator` class.

    Arguments:
        average_effect (Optional[float], optional): the average effect of the treatment. Defaults to None.
        target_col (str, optional): name of the target_col to use as the outcome. Defaults to "target".
        treatment_col (str, optional): the name of the column that contains the treatment. Defaults to "treatment".
        treatment (str, optional): name of the treatment to use as the treated group. Defaults to "B".
        scale (Optional[float], optional): the scale of the effect distribution. Defaults to None.
        range_min (float, optional): the minimum value of the target range, must be >-1.
            Defaults to -0.8, which allows for up to 5x decreases of the target.
        range_max (float, optional): the maximum value of the target range.
            Defaults to 4, which allows for up to 5x increases of the target.
        segment_cols (Optional[List[str]], optional): the columns to use for segmenting. Defaults to None.
    """

    def __init__(
        self,
        segment_cols: List[str],
        average_effect: Optional[float] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        scale: Optional[float] = None,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
    ):
        super().__init__(average_effect, target_col, treatment_col, treatment, scale)
        self._range_min = range_min or -0.8
        self._range_max = range_max or 4
        self._segment_cols = segment_cols
        self.segment_col = self._get_segment_col_name(segment_cols)

    @staticmethod
    def _get_segment_col_name(segment_cols: List[str]):
        if not isinstance(segment_cols, list):
            raise ValueError(
                f"segment_cols must be of type List[str], got type {type(segment_cols)}"
            )
        return "_cluster_" + "_".join(segment_cols)

    def _set_segment_col_values(self, df: pd.DataFrame):
        if self.segment_col in df.columns:
            raise ValueError(
                f"Cannot use {self.segment_col=} as perturbator clustering "
                f"column, as it already exists in the input dataframe!"
            )
        return df.copy().assign(
            **{self.segment_col: df[self._segment_cols].astype(str).sum(axis=1)}
        )

    def get_cluster_perturbator_fixed_params(
        self, average_effect: Optional[float] = None
    ) -> Dict[str, Any]:
        average_effect = self.get_average_effect(average_effect)
        self.check_average_effect_greater_than(average_effect, x=0)
        self.check_average_effect_less_than(average_effect, x=1)
        scale = self.get_scale(average_effect)
        return {
            "average_effect": average_effect,
            "scale": scale,
        }

    def get_cluster_perturbator(self, **kwargs) -> Perturbator:
        sampled_effect = self._sample_beta_effect(
            kwargs["average_effect"], kwargs["scale"], 1
        )
        cluster_perturbator = BetaRelativePerturbator(
            average_effect=sampled_effect,
            target_col=self.target_col,
            treatment_col=self.treatment_col,
            treatment=self.treatment,
            range_min=self._range_min,
            range_max=self._range_max,
        )
        return cluster_perturbator

    def perturbate(
        self,
        df: pd.DataFrame,
        average_effect: Optional[float] = None,
    ) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        df = self._set_segment_col_values(df)

        cluster_perturbator_params = self.get_cluster_perturbator_fixed_params(
            average_effect
        )
        df_perturbed = pd.concat(
            [
                self.get_cluster_perturbator(**cluster_perturbator_params).perturbate(
                    df=df[df[self.segment_col] == cluster].copy()
                )
                for cluster in df[self.segment_col].unique()
            ]
        )
        return df_perturbed.drop(columns=self.segment_col).reset_index(drop=True)

    @classmethod
    def from_config(cls, config):
        """Creates a Perturbator object from a PowerConfig object"""
        return cls(
            average_effect=config.average_effect,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            scale=config.scale,
            range_min=config.range_min,
            range_max=config.range_max,
            segment_cols=config.segment_cols,
        )


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
