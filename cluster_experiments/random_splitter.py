import random
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class RandomSplitter(ABC):
    """
    Abstract class to split instances in a switchback or clustered way. It can be used to create a calendar/split of clusters
    or to run a power analysis.

    In order to create your own RandomSplitter, you either write the following two methods:
    * treatment_assignment: If you are deriving from clustered or switchback splitters, no need for this. The goal of this is, given the output of sample_treatment,
    prepare such that it can be added to the dataframe by building a list of dictionaries with clusters and treatments.
    * sample_treatment: This is what needs to be implemented. It should return a list of same length as the number of clusters, with the treatment
    received to each cluster.

    Or write your own assign_treatment_df method, that takes a dataframe as an input and returns the same dataframe with the treatment_col column.

    Arguments:
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.

    """

    def __init__(
        self,
        cluster_cols: Optional[List[str]] = None,
        treatments: Optional[List[str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.cluster_cols = cluster_cols or []
        self.treatment_col = treatment_col

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
        )


class ClusteredSplitter(RandomSplitter):
    """
    Splits randomly using clusters

    Arguments:
        cluster_cols: List of columns to use as clusters
        treatments: list of treatments
        treatment_col: Name of the column with the treatment variable.

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
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.cluster_cols = cluster_cols
        self.treatment_col = treatment_col

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
        return random.choices(self.treatments, k=len(cluster_df))


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
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.treatment_col = treatment_col

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
        df[self.treatment_col] = random.choices(self.treatments, k=len(df))
        return df

    @classmethod
    def from_config(cls, config):
        """Creates a NonClusteredSplitter from a PowerConfig"""
        return cls(treatments=config.treatments, treatment_col=config.treatment_col)
