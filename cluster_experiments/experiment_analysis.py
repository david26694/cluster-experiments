from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import statsmodels.api as sm

from cluster_experiments.pre_experiment_covariates import TargetAggregation


class ExperimentAnalysis(ABC):
    def __init__(
        self,
        target_col: str,
        treatment: str,
        treatment_col: str,
        cluster: str,
        *args,
        **kwargs,
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster = cluster

    def _create_binary_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.treatment_col] = (df[self.treatment_col] == self.treatment).astype(int)
        return df

    @abstractmethod
    def get_pvalue(
        self,
        df: pd.DataFrame,
    ) -> float:
        pass


class GeeExperimentAnalysis(ExperimentAnalysis):
    def __init__(
        self,
        target_col: str,
        treatment: str,
        treatment_col: str,
        cluster: str,
        covariates: List[str],
        *args,
        **kwargs,
    ):
        super().__init__(target_col, treatment, treatment_col, cluster)
        self.covariates = covariates
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = (f"{self.target_col} ~ {' + '.join(self.regressors)}",)
        self.fam = sm.families.Gaussian()
        self.va = sm.cov_struct.Exchangeable()

    def get_pvalue(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df = self._create_binary_treatment(df)

        results_gee = sm.GEE.from_formula(
            self.formula,
            data=df,
            groups=df[self.cluster],
            family=self.fam,
            cov_struct=self.va,
        ).fit()
        return results_gee.pvalues[self.treatment_col]


class GeePreTreatmentExperimentAnalysis(GeeExperimentAnalysis):
    def __init__(
        self,
        target_col: str,
        treatment: str,
        treatment_col: str,
        cluster: str,
        aggregator: TargetAggregation,
        *args,
        **kwargs,
    ):
        # We've genertated a dependency on this name. TODO: Remove it
        covariates = [f"{self.target_col}_smooth_mean"]
        super().__init__(target_col, treatment, treatment_col, cluster, covariates)
        self.aggregator = aggregator

    def get_pvalue(self, df: pd.DataFrame) -> float:
        # Do target aggregation in here
        return super().get_pvalue(df)
