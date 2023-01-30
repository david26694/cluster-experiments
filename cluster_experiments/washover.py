import datetime
from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Washover(ABC):
    @abstractmethod
    def washover(
        self,
        df: pd.DataFrame,
        time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
    ) -> pd.DataFrame:
        pass


class EmptyWashover(Washover):
    def washover(
        self,
        df: pd.DataFrame,
        time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
    ) -> pd.DataFrame:
        return df


class ConstantWashover(Washover):
    def __init__(self, washover_time_delta: datetime.timedelta):
        self.washover_time_delta = washover_time_delta

    def washover(
        self,
        df: pd.DataFrame,
        time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
    ) -> pd.DataFrame:
        non_time_cols = list(set(cluster_cols) - set([time_col]))
        # For each cluster, we need to check if treatment has changed wrt last time

        df_agg = df.drop_duplicates(subset=cluster_cols + [treatment_col]).copy()
        df_agg["__changed"] = (
            df_agg.groupby(non_time_cols)[treatment_col].shift(1)
            != df_agg[treatment_col]
        )
        df_agg = df_agg.loc[:, cluster_cols + ["__changed"]]
        return (
            df.merge(df_agg, on=cluster_cols, how="inner")
            .assign(
                __time_since_switch=lambda x: x[f"og___{time_col}"] - x[time_col],
                __after_washover=lambda x: x["__time_since_switch"]
                > self.washover_time_delta,
            )
            # add not changed in query
            .query("__after_washover or not __changed")
            .drop(columns=["__time_since_switch", "__after_washover", "__changed"])
        )
