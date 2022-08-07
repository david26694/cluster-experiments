import pandas as pd


class Aggregator:
    def __init__(
        self,
        agg_col: str = "",
        target_col: str = "target",
        smoothing_factor: int = 20,
    ):
        self.agg_col = agg_col
        self.target_col = target_col
        self.smoothing_factor = smoothing_factor
        self.is_empty = True

    @classmethod
    def from_config(cls, config):
        return cls(
            agg_col=config.agg_col,
            target_col=config.target_col,
            smoothing_factor=config.smoothing_factor,
        )


class TargetAggregation(Aggregator):
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

    def set_pre_experiment_agg(self, pre_experiment_df: pd.DataFrame) -> None:
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

    def add_pre_experiment_agg(self, df: pd.DataFrame) -> pd.DataFrame:
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
