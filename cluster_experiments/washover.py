import datetime
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from cluster_experiments.utils import _original_time_column


class Washover(ABC):
    """Abstract class to model washovers in the switchback splitter."""

    def _validate_columns(
        self,
        df: pd.DataFrame,
        truncated_time_col: str,
        cluster_cols: List[str],
        original_time_col: str,
    ):
        """Validate that all the columns required for the washover are present in the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe.
            truncated_time_col (str): Name of the truncated time column.
            cluster_cols (List[str]): List of clusters of experiment.
            original_time_col (str): Name of the original time column.

        Returns:
            None: This method does not return any data; it only performs validation.

        """
        if original_time_col not in df.columns:
            raise ValueError(
                f"{original_time_col = } is not in the dataframe columns and/or not specified as an input."
            )
        if truncated_time_col not in cluster_cols:
            raise ValueError(f"{truncated_time_col = } is not in the cluster columns.")
        for col in cluster_cols:
            if col not in df.columns:
                raise ValueError(f"{col = } cluster is not in the dataframe columns.")

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
        ```
        """
        return df


class ConstantWashover(Washover):
    """Constant washover - we drop all rows in the washover period when
    there is a switch where the treatment is different."""

    def __init__(self, washover_time_delta: datetime.timedelta):
        self.washover_time_delta = washover_time_delta

    def washover(
        self,
        df: pd.DataFrame,
        truncated_time_col: str,
        treatment_col: str,
        cluster_cols: List[str],
        original_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Constant washover - we drop all rows in the washover period when
        there is a switch where the treatment is different.

        Args:
            df (pd.DataFrame): Input dataframe.
            truncated_time_col (str): Name of the truncated time column.
            treatment_col (str): Name of the treatment column.
            cluster_cols (List[str]): List of clusters of experiment.
            original_time_col (Optional[str], optional): Name of the original time column.

        Returns:
            pd.DataFrame: Same dataframe as input without the rows in the washover period.

        Usage:
        ```python
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta

        from cluster_experiments import ConstantWashover

        np.random.seed(42)

        num_rows = 10

        def random_timestamp(start_time, end_time):
            time_delta = end_time - start_time
            random_seconds = np.random.randint(0, time_delta.total_seconds())
            return start_time + timedelta(seconds=random_seconds)

        def generate_data(start_time, end_time, treatment):
            data = {
                'order_id': np.random.randint(10**9, 10**10, size=num_rows),
                'city_code': 'VAL',
                'activation_time_local': [random_timestamp(start_time, end_time) for _ in range(num_rows)],
                'bin_start_time_local': start_time,
                'treatment': treatment
            }
            return pd.DataFrame(data)

        start_times = [datetime(2024, 1, 22, 9, 0), datetime(2024, 1, 22, 11, 0),
                    datetime(2024, 1, 22, 13, 0), datetime(2024, 1, 22, 15, 0)]

        treatments = ['control', 'variation', 'variation', 'control']

        dataframes = [generate_data(start, start + timedelta(hours=2), treatment) for start, treatment in zip(start_times, treatments)]

        df = pd.concat(dataframes).sort_values(by='activation_time_local').reset_index(drop=True)

        ## Define washover with 30 min duration
        washover = ConstantWashover(washover_time_delta=timedelta(minutes=30))

        ## Apply washover to the dataframe, the orders with activation time within the first 30 minutes after every change in the treatment column, clustering by city and 2h time bin, will be dropped
        df_analysis_washover = washover.washover(
            df=df,
            truncated_time_col='bin_start_time_local',
            treatment_col='treatment',
            cluster_cols=['city_code','bin_start_time_local'],
            original_time_col='activation_time_local',
        )
        ```
        """
        # Set original time column
        original_time_col = (
            original_time_col
            if original_time_col
            else _original_time_column(truncated_time_col)
        )

        # Validate columns
        self._validate_columns(df, truncated_time_col, cluster_cols, original_time_col)

        # Cluster columns that do not involve time
        non_time_cols = list(set(cluster_cols) - set([truncated_time_col]))
        # For each cluster, we need to check if treatment has changed wrt last time
        df_agg = df.sort_values([original_time_col]).copy()
        df_agg = df_agg.drop_duplicates(subset=cluster_cols + [treatment_col])

        if non_time_cols:
            df_agg["__changed"] = (
                df_agg.groupby(non_time_cols)[treatment_col].shift(1)
                != df_agg[treatment_col]
            )
        else:
            df_agg["__changed"] = (
                df_agg[treatment_col].shift(1) != df_agg[treatment_col]
            )
        df_agg = df_agg.loc[:, cluster_cols + ["__changed"]]
        return (
            df.merge(df_agg, on=cluster_cols, how="inner")
            .assign(
                __time_since_switch=lambda x: x[original_time_col].astype(
                    "datetime64[ns]"
                )
                - x[truncated_time_col].astype("datetime64[ns]"),
                __after_washover=lambda x: x["__time_since_switch"]
                > self.washover_time_delta,
            )
            # add not changed in query
            .query("__after_washover or not __changed")
            .drop(columns=["__time_since_switch", "__after_washover", "__changed"])
        )

    @classmethod
    def from_config(cls, config) -> "Washover":
        if not config.washover_time_delta:
            raise ValueError(
                f"Washover time delta must be specified for ConstantWashover, while it is {config.washover_time_delta = }"
            )

        washover_time_delta = config.washover_time_delta
        if isinstance(washover_time_delta, int):
            washover_time_delta = datetime.timedelta(minutes=config.washover_time_delta)
        return cls(washover_time_delta=washover_time_delta)


class TwoEventsWashover(Washover):
    """
    For each record in our experiment, two events' timestamps,
    login_timestamp and logout_timestamp.
     We want to apply washover such that if there is a change in treatment
    between login_timestamp and logout_timestamp.
    Ex:
    Calendar:
        +-----------+-------+
        | treatment | time  |
        +-----------+-------+
        | A         | 10:00 |
        | B         | 11:00 |
        | B         | 12:00 |
        +-----------+-------+
    Record data:

        +----+------------+----------+
        | id | start_time | end_time |
        +----+------------+----------+
        |  1 | 10:50      | 10:59    |
        |  2 | 10:51      | 11:01    |
        |  3 | 11:01      | 11:05    |
        |  4 | 11:01      | 12:01    |
        |  5 | 12:01      | 12:05    |
        +----+------------+----------+

    This washover will drop record number 2.
    """

    def __init__(
        self,
        calendar_df: pd.DataFrame,
        time_column_calendar: str = "start_time",
        treatment_column_calendar: str = "treatment",
    ):
        """
        Args:
            calendar_df (pd.DataFrame): df with the calendar of treatments.
            time_column_calendar (str): Name of the time column in the calendar df.
            treatment_column_calendar (str): Name of the treatment column in the calendar df.
            record_df (pd.DataFrame): Record dataframe.
            start_time_column (str): Name of the start time column in the record dataframe.
            end_time_column (str): Name of the end time column in the record dataframe.
        """
        self.calendar_df = calendar_df
        self.time_column_calendar = time_column_calendar
        self.treatment_column_calendar = treatment_column_calendar

    def _validate_columns(
        self, record_df: pd.DataFrame, start_time_column: str, end_time_column: str
    ):
        if self.time_column_calendar not in self.calendar_df.columns:
            raise ValueError(
                f"{self.time_column_calendar = } is not in the calendar dataframe columns and/or not specified as an input."
            )
        if self.treatment_column_calendar not in self.calendar_df.columns:
            raise ValueError(
                f"{self.treatment_column_calendar = } is not in the calendar dataframe columns and/or not specified as an input."
            )
        if start_time_column not in record_df.columns:
            raise ValueError(
                f"{start_time_column = } is not in the record dataframe columns and/or not specified as an input."
            )
        if end_time_column not in record_df.columns:
            raise ValueError(
                f"{end_time_column = } is not in the record dataframe columns and/or not specified as an input."
            )

    def washover(
        self,
        record_df: pd.DataFrame,
        start_time_column: str,
        end_time_column: str,
        treatment_column: Optional[str] = None,
    ):
        """
        Return the Dataframe after applying the washover.
        """
        self._validate_columns(record_df, start_time_column, end_time_column)

        pass


# This is kept in here because of circular imports, need to rethink this
washover_mapping = {"": EmptyWashover, "constant_washover": ConstantWashover}
