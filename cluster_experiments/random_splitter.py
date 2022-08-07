# Import ABC
import random
from abc import ABC, abstractmethod
from itertools import product
from math import remainder
from typing import Dict, List, Optional

import pandas as pd


class RandomSplitter(ABC):
    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.treatments = treatments or ["A", "B"]
        self.clusters = clusters
        self.dates = dates or []
        self.cluster_mapping = cluster_mapping or {}

    def split(self) -> List[Dict[str, str]]:
        sampled_treatments = self.sample_treatment()
        return self.treatment_assignment(sampled_treatments)

    @abstractmethod
    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def sample_treatment(self, *args, **kwargs) -> List[str]:
        pass

    def assign_treatment_df(
        self,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """For clustered splitter, cluster_mapping could be {"cluster": "city"}
        for SwitchbackSplitter, cluster_mapping could be {"cluster": "city", "date": "date"}"""
        df = df.copy()
        sampled_treatments = self.sample_treatment(*args, **kwargs)
        treatments_df = pd.DataFrame(
            self.treatment_assignment(sampled_treatments)
        ).rename(columns=self.cluster_mapping)
        join_columns = list(self.cluster_mapping)
        return df.merge(treatments_df, how="left", on=join_columns)

    @classmethod
    def from_config(cls, config):
        return cls(
            clusters=config.clusters,
            treatments=config.treatments,
            dates=config.dates,
            cluster_mapping=config.cluster_mapping,
        )


class ClusteredSplitter(RandomSplitter):
    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            clusters=clusters,
            treatments=treatments,
            dates=dates,
            cluster_mapping=cluster_mapping,
        )
        if not self.cluster_mapping:
            self.cluster_mapping = {"cluster": "cluster"}

    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        """Assign a treatment to a cluster"""
        return [
            {"treatment": treatment, "cluster": cluster}
            for treatment, cluster in zip(sampled_treatments, self.clusters)
        ]

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Choose randomly a treatment for eachcluster"""
        return random.choices(self.treatments, k=len(self.clusters))


class SwitchbackSplitter(RandomSplitter):
    def __init__(
        self,
        clusters: List[str],
        treatments: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        cluster_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(
            clusters=clusters,
            treatments=treatments,
            dates=dates,
            cluster_mapping=cluster_mapping,
        )
        if not self.cluster_mapping:
            self.cluster_mapping = {"cluster": "cluster", "date": "date"}

    def treatment_assignment(
        self, sampled_treatments: List[str]
    ) -> List[Dict[str, str]]:
        """For each date, we get, on each cluster, the treatment assigned to it"""
        sampled_treatments = sampled_treatments.copy()
        output = []
        for date, cluster in product(self.dates, self.clusters):
            treatment = sampled_treatments.pop(0)
            output.append({"date": date, "cluster": cluster, "treatment": treatment})
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
        remainder_treatments = random.choices(self.treatments, k=remainder_clusters)

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
        clusters_per_treatment = int(len(self.clusters) / len(self.treatments))
        remainder_clusters = int(remainder(len(self.clusters), len(self.treatments)))
        return self.get_balanced_sample(clusters_per_treatment, remainder_clusters)


class BalancedSwitchbackSplitter(SwitchbackSplitter, BalancedClusteredSplitter):
    def sample_treatment(self, *args, **kwargs) -> List[str]:
        if len(self.clusters) * len(self.dates) < len(self.treatments):
            raise ValueError("There are more treatments than clusters and dates")

        total_switches = len(self.clusters) * len(self.dates)
        clusters_per_treatment = int(total_switches / len(self.treatments))
        remainder_clusters = int(remainder(total_switches, len(self.treatments)))
        return self.get_balanced_sample(clusters_per_treatment, remainder_clusters)
