import pandas as pd


class PreExperimentFeaturizer:
    """Empty featurizer class. Used to glue the code of TargetAggregation with PowerAnalysis.

    Each PreExperimentFeaturizer should have:
    - fit_pre_experiment_data method: Uses pre experiment data to fit some kind of model to be used as a covariate and reduce variance.
    - add_pre_experiment_data method: Uses the fitted model to add the covariate on the experiment data.

    It can add aggregates of the target in older data as a covariate, or a model (cupac) to predict the target.

    """

    def __init__(self):
        self.is_empty = True

    @classmethod
    def from_config(cls, config):
        return cls()

    def fit_pre_experiment_data(self, pre_experiment_df: pd.DataFrame) -> None:
        raise NotImplementedError("Implement this method in a subclass")

    def add_pre_experiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Implement this method in a subclass")


class TargetAggregation(PreExperimentFeaturizer):
    """Adds average of target using pre-experiment data"""

    def __init__(
        self,
        agg_col: str,
        target_col: str = "target",
        smoothing_factor: int = 20,
    ):
        """Constructor for TargetAggregation

        Args:
            agg_col: Column to group by to aggregate target
            target_col: Column to aggregate
            smoothing_factor: Smoothing factor for the smoothed mean
        Usage:
        ```python
        import pandas as pd
        from cluster_experiments.pre_experiment_covariates import TargetAggregation

        df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"], "target_col": [1, 2, 3, 4, 5, 6]})
        new_df = pd.DataFrame({"agg_col": ["a", "a", "b", "b", "c", "c"]})
        target_agg = TargetAggregation("agg_col", "target_col")
        target_agg.fit_pre_experiment_data(df)
        df_with_target_agg = target_agg.add_pre_experiment_data(new_df)
        print(df_with_target_agg)
        ```

        """
        self.agg_col = agg_col
        self.target_col = target_col
        self.smoothing_factor = smoothing_factor
        self.is_empty = False
        self.mean_target_col = f"{self.target_col}_mean"
        self.smooth_mean_target_col = f"{self.target_col}_smooth_mean"
        self.pre_experiment_agg_df = pd.DataFrame()

    def _get_pre_experiment_mean(self, pre_experiment_df: pd.DataFrame) -> float:
        return pre_experiment_df[self.target_col].mean()

    def fit_pre_experiment_data(self, pre_experiment_df: pd.DataFrame) -> None:
        """Fits "target encoder" model to pre-experiment data"""
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

    def add_pre_experiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds average target of pre-experiment data to experiment data"""
        return df.merge(self.pre_experiment_agg_df, how="left", on=self.agg_col).assign(
            **{
                self.mean_target_col: lambda x: x[self.mean_target_col].fillna(
                    self.pre_experiment_mean
                ),
                self.smooth_mean_target_col: lambda x: x[
                    self.smooth_mean_target_col
                ].fillna(self.pre_experiment_mean),
            }
        )

    @classmethod
    def from_config(cls, config):
        """Creates TargetAggregation from PowerConfig"""
        return cls(
            agg_col=config.agg_col,
            target_col=config.target_col,
            smoothing_factor=config.smoothing_factor,
        )
