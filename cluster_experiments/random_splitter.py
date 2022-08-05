# Import ABC
import random
from abc import ABC, abstractmethod
from math import remainder
from typing import Dict, List, Optional


class RandomSplitter(ABC):
    def __init__(
        self,
        treatments: List[str],
        clusters: List[str],
        dates: Optional[List[str]] = None,
        random_seed: int = 42,
    ) -> None:
        self.treatments = treatments
        self.clusters = clusters
        self.dates = dates or []
        self.random_seed = random_seed

    def split(self) -> Dict:
        sampled_treatments = self.sample_treatment()
        return self.treatment_assignment(sampled_treatments)

    @abstractmethod
    def treatment_assignment(self, sampled_treatments: List[str]) -> Dict:
        pass

    @abstractmethod
    def sample_treatment(self, *args, **kwargs) -> List[str]:
        pass


class ClusteredSplitter(RandomSplitter):
    def treatment_assignment(self, sampled_treatments: List[str]) -> Dict:
        """Assign a treatment to a cluster"""
        return dict(zip(self.clusters, sampled_treatments))

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Choose randomly a treatment for eachcluster"""
        return random.choices(self.treatments, k=len(self.clusters))


class SwitchbackSplitter(RandomSplitter):
    def treatment_assignment(self, sampled_treatments: List[str]) -> Dict:
        """For each date, we get, on each cluster, the treatment assigned to it"""
        sampled_treatments = sampled_treatments.copy()
        output_dict = {}
        for date in self.dates:
            output_dict[date] = {}
            for cluster in self.clusters:
                output_dict[date][cluster] = sampled_treatments.pop()
        return output_dict

    def sample_treatment(self, *args, **kwargs) -> List[str]:
        """Randomly assign a treatment to a cluster"""
        return random.choices(self.treatments, k=len(self.clusters) * len(self.dates))


class BalancedClusteredSplitter(ClusteredSplitter):
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
