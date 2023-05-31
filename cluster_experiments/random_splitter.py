import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd

from cluster_experiments.utils import _get_mapping_key, _original_time_column
from cluster_experiments.washover import EmptyWashover, Washover, washover_mapping


class RandomSplitter(ABC):
    """
    Abstract class to split instances in a switchback or clustered way. It can be used to create a calendar/split of clusters
    or to run a power analysis.

    In order to create your own RandomSplitter, you should write your own assign_treatment_df method, that takes a dataframe as an input and returns the same dataframe with the treatment_col column.

    Arguments:
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.
        splitter_weights: weights to use for the splitter, should have the same length as treatments, each weight should correspond to an element in treatments

    """

    def __init__(
        self,
        cluster_cols: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        splitter_weights: Optional[List[float]] = None,
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.cluster_cols = cluster_cols or []
        self.treatment_col = treatment_col
        self.splitter_weights = splitter_weights

    @abstractmethod
    def assign_treatment_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes a df, randomizes treatments and adds the treatment column to the dataframe

        Arguments:
            df: dataframe to assign treatments to
        """

    @classmethod
    def from_config(cls, config):
        """Creates a RandomSplitter from a PowerConfig"""
        return cls(
            treatments=config.treatments,
            cluster_cols=config.cluster_cols,
            treatment_col=config.treatment_col,
            splitter_weights=config.splitter_weights,
        )


class ClusteredSplitter(RandomSplitter):
    """
    Splits randomly using clusters

    Arguments:
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.
        splitter_weights: weights to use for the splitter, should have the same length as treatments, each weight should correspond to an element in treatments

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import ClusteredSplitter
    splitter = ClusteredSplitter(cluster_cols=["city"])
    df = pd.DataFrame({"city": ["A", "B", "C"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        splitter_weights: Optional[List[float]] = None,
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.cluster_cols = cluster_cols
        self.treatment_col = treatment_col
        self.splitter_weights = splitter_weights

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Takes a df, randomizes treatments and adds the treatment column to the dataframe

        Arguments:
            df: dataframe to assign treatments to
        """
        df = df.copy()

        # raise error if any nulls in cluster_cols
        if df[self.cluster_cols].isnull().values.any():
            raise ValueError(
                f"Null values found in cluster_cols: {self.cluster_cols}. "
                "Please remove nulls before running the splitter."
            )

        clusters_df = df.loc[:, self.cluster_cols].drop_duplicates()
        clusters_df[self.treatment_col] = self.sample_treatment(clusters_df)
        df = df.merge(clusters_df, on=self.cluster_cols, how="left")
        return df

    def sample_treatment(
        self,
        cluster_df: pd.DataFrame,
    ) -> List[str]:
        """
        Samples treatments for each cluster

        Arguments:
            cluster_df: dataframe to assign treatments to
        """
        return random.choices(
            self.treatments, k=len(cluster_df), weights=self.splitter_weights
        )


class SwitchbackSplitter(ClusteredSplitter):
    """
    Splits randomly using clusters and time column

    It is a clustered splitter but one of the cluster columns is obtained by truncating the time column to the switch frequency.

    Arguments:
        time_col: Name of the column with the time variable.
        switch_frequency: Frequency to switch treatments. Uses pandas frequency aliases (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases)
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.
        splitter_weights: weights to use for the splitter, should have the same length as treatments, each weight should correspond to an element in treatments

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import SwitchbackSplitter
    splitter = SwitchbackSplitter(time_col="date", switch_frequency="1D", cluster_cols=["date"])
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", "2020-01-03")})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        time_col: Optional[str] = None,
        switch_frequency: Optional[str] = None,
        cluster_cols: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        splitter_weights: Optional[List[float]] = None,
        washover: Optional[Washover] = None,
    ) -> None:
        self.time_col = time_col or "date"
        self.switch_frequency = switch_frequency or "1D"
        self.cluster_cols = cluster_cols or []
        self.treatments = treatments or ["A", "B"]
        self.treatment_col = treatment_col
        self.splitter_weights = splitter_weights
        self.washover = washover or EmptyWashover()

    def _get_time_col_cluster(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        # Given the switch frequency, truncate the time column to the switch frequency
        # Using pandas frequency aliases: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        if "W" in self.switch_frequency or "M" in self.switch_frequency:
            return df[self.time_col].dt.to_period(self.switch_frequency).dt.start_time
        return df[self.time_col].dt.floor(self.switch_frequency)

    def _prepare_switchback_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Build time_col switchback column
        # Overwriting column, this is the worst! If we use the column as a covariate, we're screwed. Needs improvement
        df[_original_time_column(self.time_col)] = df[self.time_col]
        df[self.time_col] = self._get_time_col_cluster(df)
        return df

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates the switchback column, adds it to cluster_cols and then calls ClusteredSplitter assign_treatment_df

        Arguments:
            df: dataframe to assign treatments to
        """
        df = df.copy()
        df = self._prepare_switchback_df(df)
        df = super().assign_treatment_df(df)
        df = self.washover.washover(
            df,
            truncated_time_col=self.time_col,
            treatment_col=self.treatment_col,
            cluster_cols=self.cluster_cols,
        )
        return df

    @classmethod
    def from_config(cls, config) -> "SwitchbackSplitter":
        washover_cls = _get_mapping_key(washover_mapping, config.washover)
        return cls(
            time_col=config.time_col,
            switch_frequency=config.switch_frequency,
            cluster_cols=config.cluster_cols,
            treatments=config.treatments,
            treatment_col=config.treatment_col,
            splitter_weights=config.splitter_weights,
            washover=washover_cls.from_config(config),
        )


class BalancedClusteredSplitter(ClusteredSplitter):
    """Like ClusteredSplitter, but ensures that treatments are balanced among clusters. That is, if we have
    25 clusters and 2 treatments, 13 clusters should have treatment A and 12 clusters should have treatment B."""

    def sample_treatment(
        self,
        cluster_df: pd.DataFrame,
    ) -> List[str]:
        """
        Samples treatments for each cluster

        Arguments:
            cluster_df: dataframe to assign treatments to
        """
        n_clusters = len(cluster_df)
        n_treatments = len(self.treatments)
        n_per_treatment = n_clusters // n_treatments
        n_extra = n_clusters % n_treatments
        treatments = []
        for i in range(n_treatments):
            treatments += [self.treatments[i]] * (n_per_treatment + (i < n_extra))
        random.shuffle(treatments)
        return treatments


class BalancedSwitchbackSplitter(BalancedClusteredSplitter, SwitchbackSplitter):
    """
    Like SwitchbackSplitter, but ensures that treatments are balanced among clusters. That is, if we have
    25 clusters and 2 treatments, 13 clusters should have treatment A and 12 clusters should have treatment B.
    """

    pass


class NonClusteredSplitter(RandomSplitter):
    """
    Splits randomly without clusters

    Arguments:
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import NonClusteredSplitter
    splitter = NonClusteredSplitter(
        treatments=["A", "B"],
    )
    df = pd.DataFrame({"city": ["A", "B", "C"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        splitter_weights: Optional[List[float]] = None,
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.treatment_col = treatment_col
        self.splitter_weights = splitter_weights

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        """
        Takes a df, randomizes treatments and adds the treatment column to the dataframe

        Arguments:
            df: dataframe to assign treatments to
        """
        df = df.copy()
        df[self.treatment_col] = random.choices(
            self.treatments, k=len(df), weights=self.splitter_weights
        )
        return df

    @classmethod
    def from_config(cls, config):
        """Creates a NonClusteredSplitter from a PowerConfig"""
        return cls(
            treatments=config.treatments,
            treatment_col=config.treatment_col,
            splitter_weights=config.splitter_weights,
        )


class StratifiedClusteredSplitter(RandomSplitter):
    """
    Splits randomly with clusters, ensuring a balanced allocation of treatment groups across clusters and strata.
    To be used, for example, when having days as clusters and days of the week as stratus. This splitter will make sure
    that we won't have all Sundays in treatment and no Sundays in control.

    Arguments:
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.
        strata_cols: List of columns to use as strata

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import StratifiedClusteredSplitter
    splitter = StratifiedClusteredSplitter(cluster_cols=["city"],strata_cols=["country"])
    df = pd.DataFrame({"city": ["A", "B", "C","D"], "country":["C1","C2","C2","C1"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        strata_cols: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            cluster_cols=cluster_cols,
            treatments=treatments,
            treatment_col=treatment_col,
        )
        if not strata_cols or strata_cols == [""]:
            raise ValueError(
                f"Splitter {self.__class__.__name__} requires strata_cols,"
                f" got {strata_cols = }"
            )
        self.strata_cols = strata_cols

    def assign_treatment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df_unique_shuffled = (
            df.loc[:, list(set(self.cluster_cols + self.strata_cols))]
            .drop_duplicates()
            .sample(frac=1)
            .reset_index(drop=True)
        )

        # check that, for a given cluster, there is only 1 strata
        for strata_col in self.strata_cols:
            if (
                df_unique_shuffled.groupby(self.cluster_cols)[strata_col]
                .nunique()
                .max()
                > 1
            ):
                raise ValueError(
                    f"There are multiple values in {strata_col} for the same cluster item \n"
                    "You cannot stratify on this column",
                )

        # random shuffling
        random_sorted_treatments = list(np.random.permutation(self.treatments))

        df_unique_shuffled[self.treatment_col] = (
            df_unique_shuffled.groupby(self.strata_cols, as_index=False)
            .cumcount()
            .mod(len(random_sorted_treatments))
            .map(dict(enumerate(random_sorted_treatments)))
        )

        df = df.merge(
            df_unique_shuffled, on=self.cluster_cols + self.strata_cols, how="left"
        )

        return df

    @classmethod
    def from_config(cls, config):
        """Creates a StratifiedClusteredSplitter from a PowerConfig"""
        return cls(
            treatments=config.treatments,
            cluster_cols=config.cluster_cols,
            strata_cols=config.strata_cols,
            treatment_col=config.treatment_col,
        )


class StratifiedSwitchbackSplitter(StratifiedClusteredSplitter, SwitchbackSplitter):
    """
    Splits randomly with clusters, ensuring a balanced allocation of treatment groups across clusters and strata.
    To be used, for example, when having days as clusters and days of the week as stratus. This splitter will make sure
    that we won't have all Sundays in treatment and no Sundays in control.

    It can be created using the time_col and switch_frequency arguments, just like the SwitchbackSplitter.

    Arguments:
        time_col: Name of the column with the time variable.
        switch_frequency: Frequency of the switchback. Must be a string (e.g. "1D")
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.
        splitter_weights: List of weights for the treatments. If None, all treatments will have the same weight.
        strata_cols: List of columns to use as strata

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import StratifiedSwitchbackSplitter
    splitter = StratifiedSwitchbackSplitter(time_col="date",switch_frequency="1D",strata_cols=["country"], cluster_cols=["country", "date"])
    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02", "2020-01-03","2020-01-04"], "country":["C1","C2","C2","C1"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        time_col: str = "date",
        switch_frequency: str = "1D",
        cluster_cols: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
        splitter_weights: Optional[List[float]] = None,
        washover: Optional[Washover] = None,
        strata_cols: Optional[List[str]] = None,
    ) -> None:
        # Inherit init from SwitchbackSplitter
        SwitchbackSplitter.__init__(
            self,
            time_col=time_col,
            switch_frequency=switch_frequency,
            cluster_cols=cluster_cols,
            treatments=treatments,
            treatment_col=treatment_col,
            splitter_weights=splitter_weights,
            washover=washover,
        )
        self.strata_cols = strata_cols or ["strata"]

    def assign_treatment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._prepare_switchback_df(df)
        df = StratifiedClusteredSplitter.assign_treatment_df(self, df)
        return self.washover.washover(
            df=df,
            treatment_col=self.treatment_col,
            truncated_time_col=self.time_col,
            cluster_cols=self.cluster_cols,
        )

    @classmethod
    def from_config(cls, config) -> "StratifiedSwitchbackSplitter":
        """Creates a StratifiedSwitchbackSplitter from a PowerConfig"""
        washover_cls = _get_mapping_key(washover_mapping, config.washover)
        return cls(
            treatments=config.treatments,
            cluster_cols=config.cluster_cols,
            strata_cols=config.strata_cols,
            treatment_col=config.treatment_col,
            time_col=config.time_col,
            switch_frequency=config.switch_frequency,
            splitter_weights=config.splitter_weights,
            washover=washover_cls.from_config(config),
        )


class RepeatedSampler(RandomSplitter):
    """
    Doesn't actually split the data, but repeatedly samples (i.e. duplicates) all rows for all treatments.
    This is useful for backtesting, where we assume to have access to all counterfactuals.

    Arguments:
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import RepeatedSampler
    splitter = RepeatedSampler(
        treatments=["A", "B"],
    )
    df = pd.DataFrame({"city": ["A", "B", "C"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.treatment_col = treatment_col

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df = df.copy()

        dfs = []
        for treatment in self.treatments:
            df_treat = df.copy().assign(**{self.treatment_col: treatment})
            dfs.append(df_treat)

        return pd.concat(dfs).reset_index(drop=True)

    @classmethod
    def from_config(cls, config):
        """Creates a RepeatedSampler from a PowerConfig"""
        return cls(
            treatments=config.treatments,
            treatment_col=config.treatment_col,
        )
