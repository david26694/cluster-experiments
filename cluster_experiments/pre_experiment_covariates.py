import pandas as pd


class TargetAggregation:
    def __init__(
        self,
        experiment_start_date: str,
        agg_col: str,
        target_col: str,
        date_col: str,
        smoothing_factor: int = 20,
    ):
        self.experiment_start_date = experiment_start_date
        self.agg_col = agg_col
        self.target_col = target_col
        self.date_col = date_col
        self.smoothing_factor = smoothing_factor
        self.mean_target_col = f"{self.target_col}_mean"
        self.smooth_mean_target_col = f"{self.target_col}_smooth_mean"

    def _get_pre_experiment_mean(self, df: pd.DataFrame) -> float:
        return df[self.target_col].mean()

    def pre_experiment_agg(
        self, df: pd.DataFrame, pre_experiment_mean: float
    ) -> pd.DataFrame:
        return (
            df.assign(count=1)
            .groupby(self.agg_col, as_index=False)
            .agg({self.target_col: "sum", "count": "sum"})
            .assign(
                **{
                    self.mean_target_col: lambda x: x[self.target_col] / x["count"],
                    self.smooth_mean_target_col: lambda x: (
                        x[self.target_col] + self.smoothing_factor * pre_experiment_mean
                    )
                    / (x["count"] + self.smoothing_factor),
                }
            )
        )

    def add_pre_experiment_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        pre_experiment_df = df.query(
            f"'{self.date_col}' <= '{self.experiment_start_date}'"
        )
        experiment_df = df.query(f"'{self.date_col}' > '{self.experiment_start_date}'")
        pre_experiment_mean = self._get_pre_experiment_mean(pre_experiment_df)
        pre_experiment_agg_df = self.pre_experiment_agg(
            pre_experiment_df, pre_experiment_mean
        )
        return experiment_df.merge(
            pre_experiment_agg_df, how="left", on=self.agg_col
        ).assign(
            **{
                self.mean_target_col: lambda x: x[self.mean_target_col].fillna(
                    pre_experiment_mean
                ),
                self.smooth_mean_target_col: lambda x: x[
                    self.smooth_mean_target_col
                ].fillna(pre_experiment_mean),
            }
        )
