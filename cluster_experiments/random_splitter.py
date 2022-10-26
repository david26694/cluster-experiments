import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd


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
        pass

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
        self.strata_cols = strata_cols or ["strata"]

    def assign_treatment_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df_unique_shuffled = (
            df.loc[:, self.cluster_cols + self.strata_cols]
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
                    f"There are multiple values in {strata_col} for the same cluster item"
                    "You cannot stratify on this column"
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
