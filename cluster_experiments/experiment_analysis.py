from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
import statsmodels.api as sm


class ExperimentAnalysis(ABC):
    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str,
        treatment_col: str,
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.covariates = covariates or []

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

    @classmethod
    def from_config(cls, config):
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
        )


class GeeExperimentAnalysis(ExperimentAnalysis):
    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
        )
        self.covariates = covariates or []
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"
        self.fam = sm.families.Gaussian()
        self.va = sm.cov_struct.Exchangeable()

    def _get_cluster_column(self, df: pd.DataFrame) -> pd.Series:
        """Paste all strings of cluster_cols in one single column"""
        df = df.copy()
        return df[self.cluster_cols].sum(axis=1)

    def get_pvalue(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df = self._create_binary_treatment(df)

        results_gee = sm.GEE.from_formula(
            self.formula,
            data=df,
            groups=self._get_cluster_column(df),
            family=self.fam,
            cov_struct=self.va,
        ).fit()
        return results_gee.pvalues[self.treatment_col]


class GeeExperimentAnalysisAggMean(ExperimentAnalysis):
    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        covariates = covariates or []
        covariates = covariates.copy() + [f"{self.target_col}_smooth_mean"]
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
        )
