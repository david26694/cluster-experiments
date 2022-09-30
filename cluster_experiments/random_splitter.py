import random
from abc import ABC
from itertools import product
from typing import Dict, List, Optional

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
        clusters: list of clusters to split
        treatments: list of treatments
        dates: list of dates (switches)
        cluster_mapping: dictionary to map the keys cluster and date to the actual names of the columns of the dataframe. For clustered splitter, cluster_mapping could be {"cluster": "city"}. for SwitchbackSplitter, cluster_mapping could be {"cluster": "city", "date": "date"}
        treatment_col: Name of the column with the treatment variable.

    """

    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.clusters = clusters
        self.dates = dates or []
        self.cluster_mapping = cluster_mapping or {}
        self.treatment_col = treatment_col

    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        """
        Prepares the data of the treatment assignment for the dataframe. It should take as input some list of treatments ["A", "B", "B", "A"] and return a list of dictionaries,
        where each element has information about the cluster and treatment, like {"cluster": "Cluster 1", "treatment": "A"}.
        """
        raise NotImplementedError()

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """
        Randomly samples treatments for each cluster.
        """
        raise NotImplementedError()

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Takes a df, randomizes treatments and adds the treatment column to the dataframe

        Arguments:
            df: dataframe to assign treatments to
            args: arguments to pass to sample_treatment
            kwargs: keyword arguments to pass to sample_treatment
        """
        df = df.copy()
        sampled_treatments = self.sample_treatment(*args, **kwargs)
        treatments_df = pd.DataFrame(
            self.treatment_assignment(sampled_treatments)
        ).rename(columns=self.cluster_mapping)
        join_columns = list(self.cluster_mapping.values())
        return df.merge(treatments_df, how="left", on=join_columns)

    @classmethod
    def from_config(cls, config):
        """Creates a RandomSplitter from a PowerConfig"""
        return cls(
            clusters=config.clusters,
            treatments=config.treatments,
            dates=config.dates,
            cluster_mapping=config.cluster_mapping,
            treatment_col=config.treatment_col,
        )


class ClusteredSplitter(RandomSplitter):
    """
    Splits randomly using clusters
    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import ClusteredSplitter
    splitter = ClusteredSplitter(
        clusters=["A", "B", "C"],
        treatments=["A", "B"],
        cluster_mapping={"cluster": "city"},
    )
    df = pd.DataFrame({"city": ["A", "B", "C"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        super().__init__(
            clusters=clusters,
            treatments=treatments,
            dates=dates,
            cluster_mapping=cluster_mapping,
            treatment_col=treatment_col,
        )
        if not self.cluster_mapping:
            self.cluster_mapping = {"cluster": "cluster"}

    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        """Assign each sampled treatment to a cluster"""
        return [
            {self.treatment_col: treatment, "cluster": cluster}
            for treatment, cluster in zip(sampled_treatments, self.clusters)
        ]

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Choose randomly a treatment for each cluster"""
        return random.choices(self.treatments, k=len(self.clusters))


class SwitchbackSplitter(RandomSplitter):
    """
    Splits randomly using clusters and dates

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import SwitchbackSplitter
    splitter = SwitchbackSplitter(
        clusters=["A", "B", "C"],
        treatments=["A", "B"],
        dates=["2020-01-01", "2020-01-02"],
        cluster_mapping={"cluster": "city", "date": "date"},
    )
    df = pd.DataFrame({"city": ["A", "B", "C"], "date": ["2020-01-01", "2020-01-02", "2020-01-01"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        super().__init__(
            clusters=clusters,
            treatments=treatments,
            dates=dates,
            cluster_mapping=cluster_mapping,
            treatment_col=treatment_col,
        )
        if not self.cluster_mapping:
            self.cluster_mapping = {"cluster": "cluster", "date": "date"}

        if len(self.dates) == 0:
            raise ValueError("SwitchbackSplitter requires dates")

    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        """For each date, we get, on each cluster, the treatment assigned to it"""
        sampled_treatments = sampled_treatments.copy()
        output = []
        for date, cluster in product(self.dates, self.clusters):
            treatment = sampled_treatments.pop(0)
            output.append(
                {"date": date, "cluster": cluster, self.treatment_col: treatment}
            )
        return output

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Randomly assign a treatment to a cluster"""
        return random.choices(self.treatments, k=len(self.clusters) * len(self.dates))


class BalancedClusteredSplitter(ClusteredSplitter):
    """Like ClusteredSplitter, but ensures that treatments are balanced among clusters. That is, if we have
    25 clusters and 2 treatments, 13 clusters should have treatment A and 12 clusters should have treatment B."""

    def get_balanced_sample(
        self, clusters_per_treatment: int, remainder_clusters: int
    ) -> List[str]:
        """Given the number of clusters per treatment and the remainder clusters (results
        of the integer division between the number of clusters and the number of treatments),
        obtain, in the most balanced way, a list of treatments such that the difference between
        the number of clusters per treatment is minimal among clusters"""
        remainder_treatments = random.sample(self.treatments, k=remainder_clusters)

        sampled_treatments = []
        for treatment in self.treatments:
            sampled_treatments.extend([treatment] * clusters_per_treatment)
        sampled_treatments.extend(remainder_treatments)
        random.shuffle(sampled_treatments)
        return sampled_treatments

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Randomly assign a treatment to a cluster"""
        if len(self.clusters) < len(self.treatments):
            raise ValueError("There are more treatments than clusters")
        clusters_per_treatment = len(self.clusters) // len(self.treatments)
        remainder_clusters = len(self.clusters) % len(self.treatments)
        return self.get_balanced_sample(clusters_per_treatment, remainder_clusters)


class BalancedSwitchbackSplitter(SwitchbackSplitter, BalancedClusteredSplitter):
    """
    Like SwitchbackSplitter, but ensures that treatments are balanced among clusters.

    Usage:
    ```python
    import pandas as pd
    from cluster_experiments.random_splitter import BalancedSwitchbackSplitter
    splitter = BalancedSwitchbackSplitter(
        clusters=["A", "B", "C"],
        treatments=["A", "B"],
        dates=["2020-01-01", "2020-01-02"],
        cluster_mapping={"cluster": "city", "date": "date"},
    )
    df = pd.DataFrame({"city": ["A", "B", "C"], "date": ["2020-01-01", "2020-01-02", "2020-01-01"]})
    df = splitter.assign_treatment_df(df)
    print(df)
    ```
    """

    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
        treatment_col: str = "treatment",
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.clusters = clusters
        self.dates = dates or []
        self.cluster_mapping = cluster_mapping or {}
        self.treatment_col = treatment_col

        if not self.cluster_mapping:
            self.cluster_mapping = {"cluster": "cluster", "date": "date"}

        if len(self.dates) == 0:
            raise ValueError("SwitchbackSplitter requires dates")

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        if len(self.clusters) * len(self.dates) < len(self.treatments):
            raise ValueError("There are more treatments than clusters and dates")

        total_switches = len(self.clusters) * len(self.dates)
        clusters_per_treatment = total_switches // len(self.treatments)
        remainder_clusters = total_switches % len(self.treatments)
        return self.get_balanced_sample(clusters_per_treatment, remainder_clusters)


class NonClusteredSplitter(RandomSplitter):
    """
    Splits randomly without clusters
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
        *args,
        **kwargs,
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
