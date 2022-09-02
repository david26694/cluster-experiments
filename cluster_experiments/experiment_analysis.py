from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd
import statsmodels.api as sm


class ExperimentAnalysis(ABC):
    """Abstract class to run the analysis of a given experiment"""

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        """
        Creates an object to run the analysis of a given experiment, after the data is collected.
        It can also be used as a component of the PowerAnalysis class.

        Arguments:
            cluster_cols: list of columns to use as clusters
            target_col: name of the column containing the variable to measure
            treatment_col: name of the column containing the treatment variable
            treatment: name of the treatment to use as the treated group
            covariates: list of columns to use as covariates
        """
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.covariates = covariates or []

    def _get_cluster_column(self, df: pd.DataFrame) -> pd.Series:
        """Paste all strings of cluster_cols in one single column"""
        df = df.copy()
        return df[self.cluster_cols].sum(axis=1)

    def _create_binary_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms treatment column into 0 - 1 column"""
        df = df.copy()
        df[self.treatment_col] = (df[self.treatment_col] == self.treatment).astype(int)
        return df

    @abstractmethod
    def get_pvalue(
        self,
        df: pd.DataFrame,
    ) -> float:
        """Returns the p-value of the analysis"""
        pass

    @classmethod
    def from_config(cls, config):
        """Creates an ExperimentAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
        )


class GeeExperimentAnalysis(ExperimentAnalysis):
    """Class to run GEE analysis"""

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        """
        Usage:

        ```python
        from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
        import pandas as pd

        df = pd.DataFrame({
            'x': [1, 2, 3, 0, 0, 1],
            'treatment': ["A"] * 3 + ["B"] * 3,
            'cluster': [1] * 6,
        })

        GeeExperimentAnalysis(
            cluster_cols=['cluster'],
            target_col='x',
        ).get_pvalue(df)
        ```
        """
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
        )
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"
        self.fam = sm.families.Gaussian()
        self.va = sm.cov_struct.Exchangeable()

    def get_pvalue(self, df: pd.DataFrame) -> float:
        """Returns the p-value of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
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


class GeeExperimentAnalysisAggMean(GeeExperimentAnalysis):
    # TODO: Should we drop this class?
    """Class to run GEE analysis with aggregated mean as a covariate"""

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
    ):
        covariates = covariates or []
        covariates = covariates.copy() + [f"{target_col}_smooth_mean"]
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
        )
