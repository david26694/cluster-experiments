import datetime
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from cluster_experiments.utils import _original_time_column


class Washover(ABC):
    """Abstract class to model washovers in the switchback splitter."""

    @abstractmethod
    def washover(
        self,
        df: pd.DataFrame,
        truncated_time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
        original_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Abstract method to add washvover to the dataframe."""

    @classmethod
    def from_config(cls, config) -> "Washover":
        return cls()


class EmptyWashover(Washover):
    """No washover - assumes no spill-over effects from one treatment to another."""

    def washover(
        self,
        df: pd.DataFrame,
        truncated_time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
        original_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """No washover - returns the same dataframe as input.

        Args:
            df (pd.DataFrame): Input dataframe.
            truncated_time_col (str): Name of the truncated time column.
            treatment_col (str): Name of the treatment column.
            cluster_cols (List[str]): List of clusters of experiment.
            original_time_col (Optional[str], optional): Name of the original time column.

        Returns:
            pd.DataFrame: Same dataframe as input.

        Usage:
        ```python
        from cluster_experiments import SwitchbackSplitter
        from cluster_experiments import EmptyWashover

        washover = EmptyWashover()

        n = 10
        df = pd.DataFrame(
            {
                # Random time each minute in 2022-01-01, length 10
                "time": pd.date_range("2022-01-01", "2022-01-02", freq="1min")[
                    np.random.randint(24 * 60, size=n)
                ],
                "city": random.choices(["TGN", "NYC", "LON", "REU"], k=n),
            }
        )


        splitter = SwitchbackSplitter(
            washover=washover,
            time_col="time",
            cluster_cols=["city", "time"],
            treatment_col="treatment",
            switch_frequency="30T",
        )

        out_df = splitter.assign_treatment_df(df=washover_split_df)

        """
        return df


class TwoSidedWashover(Washover):
    """Two sided washover - we drop all rows before and after the switch within
    the time deltas when there is a switch where the treatment is different."""

    def __init__(
        self,
        washover_time_delta_before: datetime.timedelta,
        washover_time_delta_after: datetime.timedelta,
    ):
        self.washover_time_delta_before = washover_time_delta_before
        self.washover_time_delta_after = washover_time_delta_after

    def washover(
        self,
        df: pd.DataFrame,
        truncated_time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
        original_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Two sided washover - removes rows around the switch.

        Args:
            df (pd.DataFrame): Input dataframe.
            truncated_time_col (str): Name of the truncated time column.
            treatment_col (str): Name of the treatment column.
            cluster_cols (List[str]): List of clusters of experiment.
            original_time_col (Optional[str], optional): Name of the original time column.

        Returns:
            pd.DataFrame: Same dataframe as input.

        Usage:
        ```python
        from cluster_experiments import SwitchbackSplitter
        from cluster_experiments import ConstantWashover

        washover = TwoSidedWashover(washover_time_delta=datetime.timedelta(minutes=30))

        n = 10
        df = pd.DataFrame(
            {
                # Random time each minute in 2022-01-01, length 10
                "time": pd.date_range("2022-01-01", "2022-01-02", freq="1min")[
                    np.random.randint(24 * 60, size=n)
                ],
                "city": random.choices(["TGN", "NYC", "LON", "REU"], k=n),
            }
        )


        splitter = SwitchbackSplitter(
            washover=washover,
            time_col="time",
            cluster_cols=["city", "time"],
            treatment_col="treatment",
            switch_frequency="30T",
        )

        out_df = splitter.assign_treatment_df(df=washover_split_df)

        """
        # Set original time column
        original_time_col = (
            original_time_col
            if original_time_col
            else _original_time_column(truncated_time_col)
        )

        # Cluster columns that do not involve time
        non_time_cols = list(set(cluster_cols) - set([truncated_time_col]))

        # For each cluster, we need to check if treatment has changed wrt last time
        df_agg = df.drop_duplicates(subset=cluster_cols + [treatment_col]).copy()
        df_agg["__changed"] = (
            df_agg.groupby(non_time_cols)[treatment_col].shift(1)
            != df_agg[treatment_col]
        ) & df_agg.groupby(non_time_cols)[treatment_col].shift(1).notnull()

        # We also check if treatment changes for the next time
        df_agg["__changed_next"] = (
            df_agg.groupby(non_time_cols)[treatment_col].shift(-1)
            != df_agg[treatment_col]
        ) & df_agg.groupby(non_time_cols)[treatment_col].shift(-1).notnull()

        # Calculate switch start of the next time
        df_agg[f"__next_{truncated_time_col}"] = df_agg.groupby(non_time_cols)[
            truncated_time_col
        ].shift(-1)

        # Clean switch df
        df_agg = df_agg.loc[
            :,
            cluster_cols
            + ["__changed", "__changed_next", f"__next_{truncated_time_col}"],
        ]
        return (
            df.merge(df_agg, on=cluster_cols, how="inner")
            .assign(
                __time_since_switch=lambda x: x[original_time_col].astype(
                    "datetime64[ns]"
                )
                - x[truncated_time_col].astype("datetime64[ns]"),
                __time_to_next_switch=lambda x: x[
                    f"__next_{truncated_time_col}"
                ].astype("datetime64[ns]")
                - x[original_time_col].astype("datetime64[ns]"),
                __after_washover=lambda x: (
                    x["__time_since_switch"] > self.washover_time_delta_after
                ),
                __before_washover=lambda x: (
                    x["__time_to_next_switch"] > self.washover_time_delta_before
                ),
            )
            # if no change or too late after change, don't drop
            .query("__after_washover or not __changed")
            # if no change to next switch or too early before change, don't drop
            .query("__before_washover or not __changed_next")
            .drop(
                columns=[
                    "__time_since_switch",
                    "__time_to_next_switch",
                    "__after_washover",
                    "__before_washover",
                    "__changed",
                    "__changed_next",
                    f"__next_{truncated_time_col}",
                ]
            )
        )

    @classmethod
    def from_config(cls, config) -> "Washover":
        if (
            not config.washover_time_delta_before
            or not config.washover_time_delta_after
        ):
            raise ValueError(
                f"Washover time deltas must be specified for , while it is {config.washover_time_delta_before = } and {config.washover_time_delta_after = }"
            )
        return cls(
            washover_time_delta_before=config.washover_time_delta,
            washover_time_delta_after=config.washover_time_delta,
        )


class ConstantWashover(TwoSidedWashover):
    """Constant washover - we drop all rows in the washover period after
    the switch when the treatment is different."""

    def __init__(self, washover_time_delta: datetime.timedelta):
        super().__init__(datetime.timedelta(seconds=0), washover_time_delta)

    @classmethod
    def from_config(cls, config) -> "Washover":
        if not config.washover_time_delta:
            raise ValueError(
                f"Washover time delta must be specified for SimetricWashover, while it is {config.washover_time_delta = }"
            )
        return cls(washover_time_delta=config.washover_time_delta)


class SimmetricWashover(TwoSidedWashover):
    """Simmetric washover - we drop all rows in the washover period before
    and after the switch when the treatment is different."""

    def __init__(self, washover_time_delta: datetime.timedelta):
        super().__init__(
            washover_time_delta_before=washover_time_delta,
            washover_time_delta_after=washover_time_delta,
        )

    @classmethod
    def from_config(cls, config) -> "Washover":
        if not config.washover_time_delta:
            raise ValueError(
                f"Washover time delta must be specified for SimetricWashover, while it is {config.washover_time_delta = }"
            )
        return cls(washover_time_delta=config.washover_time_delta)


# This is kept in here because of circular imports, need to rethink this
washover_mapping = {
    "": EmptyWashover,
    "constant_washover": ConstantWashover,
    "two_sided_washover": TwoSidedWashover,
    "simmetric_washover": SimmetricWashover,
}
