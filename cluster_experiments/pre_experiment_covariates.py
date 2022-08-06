import pandas as pd


class TargetAggregation:
    # TODO: df doesn't need to be passed in here, it needs to be passed in add_pre_experiment_agg
    def __init__(
        self,
        df: pd.DataFrame,
        experiment_start_date: str,
        agg_col: str,
        target_col: str,
        date_col: str,
        smoothing_factor: int = 20,
    ):
        self.df = df
        self.experiment_start_date = experiment_start_date
        self.agg_col = agg_col
        self.target_col = target_col
        self.smoothing_factor = smoothing_factor
        self.pre_experiment_df = df.query(f"'{date_col}' <= '{experiment_start_date}'")
        self.experiment_df = df.query(f"'{date_col}' > '{experiment_start_date}'")
        self.pre_experiment_mean = self.pre_experiment_df[self.target_col].mean()
        self.mean_target_col = f"{self.target_col}_mean"
        self.smooth_mean_target_col = f"{self.target_col}_smooth_mean"

    def pre_experiment_agg(self) -> pd.DataFrame:
        return (
            self.pre_experiment_df.assign(count=1)
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
        )

    def add_pre_experiment_agg(self) -> pd.DataFrame:
        pre_experiment_agg_df = self.pre_experiment_agg()
        return self.experiment_df.merge(
            pre_experiment_agg_df, how="left", on=self.agg_col
        ).assign(
            **{
                self.mean_target_col: lambda x: x[self.mean_target_col].fillna(
                    self.pre_experiment_mean
                ),
                self.smooth_mean_target_col: lambda x: x[
                    self.smooth_mean_target_col
                ].fillna(self.pre_experiment_mean),
            }
        )
