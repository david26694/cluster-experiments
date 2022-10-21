import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin


class EmptyRegressor(BaseEstimator, RegressorMixin):
    """
    Empty regressor class. It does not do anything, used to glue the code of other estimators and PowerAnalysis

    Each Regressor should have:
    - fit method: Uses pre experiment data to fit some kind of model to be used as a covariate and reduce variance.
    - predict method: Uses the fitted model to add the covariate on the experiment data.

    It can add aggregates of the target in older data as a covariate, or a model (cupac) to predict the target.
    """

    @classmethod
    def from_config(cls, config):
        return cls()


class TargetAggregation(BaseEstimator, RegressorMixin):
    """
    Adds average of target using pre-experiment data

    Args:
        agg_col: Column to group by to aggregate target
        target_col: Column to aggregate
        smoothing_factor: Smoothing factor for the smoothed mean
    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.cupac import TargetAggregation

    df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"], "target_col": [1, 2, 3, 4, 5, 6]})
    new_df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"]})
    target_agg = TargetAggregation("agg_col", "target_col")
    target_agg.fit(df.drop(columns="target_col"), df["target_col"])
    df_with_target_agg = target_agg.predict(new_df)
    print(df_with_target_agg)
    ```
    """

    def __init__(
        self,
        agg_col: str,
        target_col: str = "target",
        smoothing_factor: int = 20,
    ):
        self.agg_col = agg_col
        self.target_col = target_col
        self.smoothing_factor = smoothing_factor
        self.is_empty = False
        self.mean_target_col = f"{self.target_col}_mean"
        self.smooth_mean_target_col = f"{self.target_col}_smooth_mean"
        self.pre_experiment_agg_df = pd.DataFrame()

    def _get_pre_experiment_mean(self, pre_experiment_df: pd.DataFrame) -> float:
        return pre_experiment_df[self.target_col].mean()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetAggregation":
        """Fits "target encoder" model to pre-experiment data"""
        pre_experiment_df = X.copy()
        pre_experiment_df[self.target_col] = y

        self.pre_experiment_mean = pre_experiment_df[self.target_col].mean()
        self.pre_experiment_agg_df = (
            pre_experiment_df.assign(count=1)
            .groupby(self.agg_col, as_index=False)
            .agg({self.target_col: "sum", "count": "sum"})
            .assign(
                **{
                    self.mean_target_col: lambda x: x[self.target_col] / x["count"],
                    self.smooth_mean_target_col: lambda x: (
                        x[self.target_col]
                        + self.smoothing_factor * self.pre_experiment_mean
                    )
                    / (x["count"] + self.smoothing_factor),
                }
            )
            .drop(columns=["count", self.target_col])
        )
        return self

    def predict(self, X: pd.DataFrame) -> ArrayLike:
        """Adds average target of pre-experiment data to experiment data"""
        return (
            X.merge(self.pre_experiment_agg_df, how="left", on=self.agg_col)[
                self.smooth_mean_target_col
            ]
            .fillna(self.pre_experiment_mean)
            .values
        )

    @classmethod
    def from_config(cls, config):
        """Creates TargetAggregation from PowerConfig"""
        return cls(
            agg_col=config.agg_col,
            target_col=config.target_col,
            smoothing_factor=config.smoothing_factor,
        )
