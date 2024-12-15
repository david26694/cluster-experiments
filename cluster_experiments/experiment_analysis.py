import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm, ttest_ind, ttest_rel

from cluster_experiments.synthetic_control_utils import get_w
from cluster_experiments.utils import HypothesisEntries


@dataclass
class ConfidenceInterval:
    """
    Class to define the structure of a confidence interval.
    """

    lower: float
    upper: float
    alpha: float


@dataclass
class InferenceResults:
    """
    Class to define the structure of complete statistical analysis results.
    """

    ate: float
    p_value: float
    std_error: float
    conf_int: ConfidenceInterval


class ExperimentAnalysis(ABC):
    """
    Abstract class to run the analysis of a given experiment

    In order to create your own ExperimentAnalysis,
    you should create a derived class that implements the analysis_pvalue method.

    It can also be used as a component of the PowerAnalysis class.

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.covariates = covariates or []
        self.hypothesis = hypothesis

    def _get_cluster_column(self, df: pd.DataFrame) -> pd.Series:
        """Paste all strings of cluster_cols in one single column"""
        df = df.copy()
        return df[self.cluster_cols].astype(str).sum(axis=1)

    def _create_binary_treatment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms treatment column into 0 - 1 column"""
        df = df.copy()
        df[self.treatment_col] = (df[self.treatment_col] == self.treatment).astype(int)
        return df

    @abstractmethod
    def analysis_pvalue(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the p-value of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """

    def analysis_point_estimate(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the point estimate of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError("Point estimate not implemented for this analysis")

    def analysis_standard_error(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
    ) -> float:
        """
        Returns the standard error of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError("Standard error not implemented for this analysis")

    def analysis_confidence_interval(
        self,
        df: pd.DataFrame,
        alpha: float,
        verbose: bool = False,
    ) -> ConfidenceInterval:
        """
        Returns the confidence interval of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError(
            "Confidence Interval not implemented for this analysis"
        )

    def analysis_inference_results(
        self,
        df: pd.DataFrame,
        alpha: float,
        verbose: bool = False,
    ) -> InferenceResults:
        """
        Returns the InferenceResults object of the analysis. Expects treatment to be 0-1 variable
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        raise NotImplementedError(
            "Inference results are not implemented for this analysis"
        )

    def _data_checks(self, df: pd.DataFrame) -> None:
        """Checks that the data is correct"""
        if df[self.target_col].isnull().any():
            raise ValueError(
                f"There are null values in outcome column {self.treatment_col}"
            )

        if not is_numeric_dtype(df[self.target_col]):
            raise ValueError(
                f"Outcome column {self.target_col} should be numeric and not {df[self.target_col].dtype}"
            )

    def get_pvalue(self, df: pd.DataFrame) -> float:
        """Returns the p-value of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_pvalue(df)

    def get_point_estimate(self, df: pd.DataFrame) -> float:
        """Returns the point estimate of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_point_estimate(df)

    def get_standard_error(self, df: pd.DataFrame) -> float:
        """Returns the standard error of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_standard_error(df)

    def get_confidence_interval(
        self, df: pd.DataFrame, alpha: float
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis

        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_confidence_interval(df, alpha)

    def get_inference_results(self, df: pd.DataFrame, alpha: float) -> InferenceResults:
        """Returns the inference results of the analysis

        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_inference_results(df, alpha)

    def pvalue_based_on_hypothesis(
        self, model_result
    ) -> float:  # todo add typehint statsmodels result
        """Returns the p-value of the analysis
        Arguments:
            model_result: statsmodels result object
            verbose (Optional): bool, prints the regression summary if True

        """
        treatment_effect = model_result.params[self.treatment_col]
        p_value = model_result.pvalues[self.treatment_col]

        if HypothesisEntries(self.hypothesis) == HypothesisEntries.LESS:
            return p_value / 2 if treatment_effect <= 0 else 1 - p_value / 2
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.GREATER:
            return p_value / 2 if treatment_effect >= 0 else 1 - p_value / 2
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.TWO_SIDED:
            return p_value
        raise ValueError(f"{self.hypothesis} is not a valid HypothesisEntries")

    def _split_pre_experiment_df(self, df: pd.DataFrame):
        raise NotImplementedError(
            "This method should be implemented in the child class"
        )

    @classmethod
    def from_config(cls, config):
        """Creates an ExperimentAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
            hypothesis=config.hypothesis,
        )


class GeeExperimentAnalysis(ExperimentAnalysis):
    """
    Class to run GEE clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

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

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
        )
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"
        self.fam = sm.families.Gaussian()
        self.va = sm.cov_struct.Exchangeable()

    def fit_gee(self, df: pd.DataFrame) -> sm.GEE:
        """Returns the fitted GEE model"""
        return sm.GEE.from_formula(
            self.formula,
            data=df,
            groups=self._get_cluster_column(df),
            family=self.fam,
            cov_struct=self.va,
        ).fit()

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        if verbose:
            print(results_gee.summary())

        p_value = self.pvalue_based_on_hypothesis(results_gee)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        return results_gee.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        return results_gee.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_gee.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_gee.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_gee = self.fit_gee(df)

        std_error = results_gee.bse[self.treatment_col]
        ate = results_gee.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_gee)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_gee.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_gee.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
        )


class ClusteredOLSAnalysis(ExperimentAnalysis):
    """
    Class to run OLS clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1, 2, 0],
        'treatment': ["A"] * 2 + ["B"] * 2 + ["A"] * 2 + ["B"] * 2,
        'cluster': [1, 1, 2, 2, 3, 3, 4, 4],
    })

    ClusteredOLSAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
        )
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"
        self.cov_type = "cluster"

    def fit_ols_clustered(self, df: pd.DataFrame):
        """Returns the fitted OLS model"""
        return sm.OLS.from_formula(self.formula, data=df,).fit(
            cov_type=self.cov_type,
            cov_kwds={"groups": self._get_cluster_column(df)},
        )

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols_clustered(df)
        if verbose:
            print(results_ols.summary())

        p_value = self.pvalue_based_on_hypothesis(results_ols)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols_clustered(df)
        return results_ols.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols_clustered(df)
        return results_ols.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols_clustered(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols_clustered(df)

        std_error = results_ols.bse[self.treatment_col]
        ate = results_ols.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_ols)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
        )


class TTestClusteredAnalysis(ExperimentAnalysis):
    """
    Class to run T-test analysis on aggregated data

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import TTestClusteredAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 0, 0, 1, 1],
        'treatment': ["A", "B", "A", "B"] * 2,
        'cluster': [1, 2, 3, 4, 1, 2, 3, 4],
    })

    TTestClusteredAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """

        df_grouped = df.groupby(
            self.cluster_cols + [self.treatment_col], as_index=False
        )[self.target_col].mean()

        treatment_data = df_grouped.query(f"{self.treatment_col} == 1")[self.target_col]
        control_data = df_grouped.query(f"{self.treatment_col} == 0")[self.target_col]
        assert len(treatment_data), "treatment data should have more than 1 cluster"
        assert len(control_data), "control data should have more than 1 cluster"
        t_test_results = ttest_ind(
            treatment_data, control_data, equal_var=False, alternative=self.hypothesis
        )
        return t_test_results.pvalue

    @classmethod
    def from_config(cls, config):
        """Creates a TTestClusteredAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            hypothesis=config.hypothesis,
        )


class PairedTTestClusteredAnalysis(ExperimentAnalysis):
    """
    Class to run paired T-test analysis on aggregated data

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        strata_cols: list of index columns for paired t test. Should be a subset or equal to cluster_cols
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import PairedTTestClusteredAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 0, 0, 1, 1],
        'treatment': ["A", "B", "A", "B"] * 2,
        'cluster': [1, 2, 3, 4, 1, 2, 3, 4],
    })

    PairedTTestClusteredAnalysis(
        cluster_cols=['cluster'],
        strata_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        strata_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
    ):
        self.strata_cols = strata_cols
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis

    def _preprocessing(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        df_grouped = df.groupby(
            self.cluster_cols + [self.treatment_col], as_index=False
        )[self.target_col].mean()

        n_control = df_grouped[self.treatment_col].value_counts()[0]
        n_treatment = df_grouped[self.treatment_col].value_counts()[1]

        if n_control != n_treatment:
            logging.warning(
                f"groups don't have same number of observations, {n_treatment =} and  {n_control =}"
            )

        assert all(
            [x in self.cluster_cols for x in self.strata_cols]
        ), f"strata should be a subset or equal to cluster_cols ({self.cluster_cols = }, {self.strata_cols = })"

        df_pivot = df_grouped.pivot_table(
            columns=self.treatment_col,
            index=self.strata_cols,
            values=self.target_col,
        )

        if df_pivot.isna().sum().sum() > 0:
            logging.warning(
                f"There are missing pairs for some clusters, removing the lonely ones: {df_pivot[df_pivot.isna().any(axis=1)].to_dict()}"
            )

        if verbose:
            print(f"performing paired t test in this data \n {df_pivot} \n")

        df_pivot = df_pivot.dropna()

        return df_pivot

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the extra info if True
        """
        assert (
            type(self.cluster_cols) is list
        ), "cluster_cols needs to be a list of strings (even with one element)"
        assert (
            type(self.strata_cols) is list
        ), "strata_cols needs to be a list of strings (even with one element)"

        df_pivot = self._preprocessing(df=df)

        t_test_results = ttest_rel(
            df_pivot.iloc[:, 0], df_pivot.iloc[:, 1], alternative=self.hypothesis
        )

        if verbose:
            print(f"paired t test results: \n {t_test_results} \n")

        return t_test_results.pvalue

    @classmethod
    def from_config(cls, config):
        """Creates a PairedTTestClusteredAnalysis object from a PowerConfig object"""
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            strata_cols=config.strata_cols,
            hypothesis=config.hypothesis,
        )


class OLSAnalysis(ExperimentAnalysis):
    """
    Class to run OLS analysis

    Arguments:
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import OLSAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1],
        'treatment': ["A"] * 3 + ["B"] * 3,
    })

    OLSAnalysis(
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.covariates = covariates or []
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"
        self.hypothesis = hypothesis

    def fit_ols(self, df: pd.DataFrame):
        """Returns the fitted OLS model"""
        return sm.OLS.from_formula(self.formula, data=df).fit()

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        if verbose:
            print(results_ols.summary())

        p_value = self.pvalue_based_on_hypothesis(results_ols)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        return results_ols.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df=df)
        return results_ols.bse[self.treatment_col]

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        """Returns the confidence interval of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df)
        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound, alpha=alpha)

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        """Returns the inference results of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            alpha: significance level
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = self.fit_ols(df)

        std_error = results_ols.bse[self.treatment_col]
        ate = results_ols.params[self.treatment_col]
        p_value = self.pvalue_based_on_hypothesis(results_ols)

        # Extract the confidence interval for the treatment column
        conf_int_df = results_ols.conf_int(alpha=alpha)
        lower_bound, upper_bound = conf_int_df.loc[self.treatment_col]

        if verbose:
            print(results_ols.summary())

        # Return the confidence interval
        return InferenceResults(
            ate=ate,
            p_value=p_value,
            std_error=std_error,
            conf_int=ConfidenceInterval(
                lower=lower_bound, upper=upper_bound, alpha=alpha
            ),
        )

    @classmethod
    def from_config(cls, config):
        """Creates an OLSAnalysis object from a PowerConfig object"""
        return cls(
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            covariates=config.covariates,
            hypothesis=config.hypothesis,
        )


class MLMExperimentAnalysis(ExperimentAnalysis):
    """
    Class to run Mixed Linear Models clustered analysis

    Arguments:
        cluster_cols: list of columns to use as clusters
        target_col: name of the column containing the variable to measure
        treatment_col: name of the column containing the treatment variable
        treatment: name of the treatment to use as the treated group
        covariates: list of columns to use as covariates
        hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis

    Usage:

    ```python
    from cluster_experiments.experiment_analysis import MLMExperimentAnalysis
    import pandas as pd

    df = pd.DataFrame({
        'x': [1, 2, 3, 0, 0, 1],
        'treatment': ["A"] * 3 + ["B"] * 3,
        'cluster': [1] * 6,
    })

    MLMExperimentAnalysis(
        cluster_cols=['cluster'],
        target_col='x',
    ).get_pvalue(df)
    ```
    """

    def __init__(
        self,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
    ):
        super().__init__(
            target_col=target_col,
            treatment_col=treatment_col,
            cluster_cols=cluster_cols,
            treatment=treatment,
            covariates=covariates,
            hypothesis=hypothesis,
        )
        self.regressors = [self.treatment_col] + self.covariates
        self.formula = f"{self.target_col} ~ {' + '.join(self.regressors)}"

        self.re_formula = None
        self.vc_formula = None

    def fit_mlm(self, df: pd.DataFrame) -> sm.MixedLM:
        """Returns the fitted MLM model"""
        return sm.MixedLM.from_formula(
            formula=self.formula,
            data=df,
            groups=self._get_cluster_column(df),
            re_formula=self.re_formula,
            vc_formula=self.vc_formula,
        ).fit()

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        if verbose:
            print(results_mlm.summary())

        p_value = self.pvalue_based_on_hypothesis(results_mlm)
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        return results_mlm.params[self.treatment_col]

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_mlm = self.fit_mlm(df)
        return results_mlm.bse[self.treatment_col]


class SyntheticControlAnalysis(ExperimentAnalysis):
    """
    Class to run Synthetic control analysis. It expects only one treatment cluster.

    Arguments:

        target_col (str): The name of the column containing the variable to measure.
        treatment_col (str): The name of the column containing the treatment variable.
        treatment (str): The name of the treatment to use as the treated group.
        cluster_cols (list): A list of columns to use as clusters.
        hypothesis (str): One of "two-sided", "less", "greater" indicating the hypothesis.
        time_col (str): The name of the column containing the time data.
        intervention_date (str): The date when the intervention occurred.
    Usage:

    ```python
    from cluster_experiments.experiment_analysis import SyntheticControlAnalysis
    import pandas as pd
    import numpy as np
    from itertools import product

    dates = pd.date_range("2022-01-01", "2022-01-31", freq="d")

    users = [f"User {i}" for i in range(10)]

    # Create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values

    df["treatment"] = "A"
    df.loc[(df["user"] == "User 5"), "treatment"] = "B"

    SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", intervention_date="2022-01-15"
    ).get_pvalue(df)

    ```
    """

    def __init__(
        self,
        intervention_date: str,
        cluster_cols: List[str],
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
        time_col: str = "date",
    ):
        super().__init__(
            treatment=treatment,
            target_col=target_col,
            treatment_col=treatment_col,
            hypothesis=hypothesis,
            cluster_cols=cluster_cols,
        )

        self.time_col = time_col
        self.intervention_date = intervention_date

        if time_col in cluster_cols:
            raise ValueError("time columns should not be in cluster columns")

    def _fit(self, pre_experiment_df: pd.DataFrame, verbose: bool) -> np.ndarray:
        """Returns the weight of each donor"""

        if not any(pre_experiment_df[self.treatment_col] == 1):
            raise ValueError("No treatment unit found in the data.")

        X = (
            pre_experiment_df.query(f"{self.treatment_col} == 0")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T
        )

        y = (
            pre_experiment_df.query(f"{self.treatment_col} == 1")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T.iloc[:, 0]
        )

        weights = get_w(X, y, verbose)

        return weights

    def _predict(
        self, df: pd.DataFrame, weights: np.ndarray, treatment_cluster: str
    ) -> pd.DataFrame:
        """
        This method adds a column with the synthetic results and filter only the treatment unit.

        First, it calculates the weights of each donor in the control group using the `fit_synthetic` method.
        It then uses these weights to create a synthetic control group that closely matches the treatment unit before the intervention.
        The synthetic control group is added to the treatment unit in the dataframe.
        """
        synthetic = (
            df[self._get_cluster_column(df) != treatment_cluster]
            .pivot(index=self.time_col, columns=self.cluster_cols)[self.target_col]
            .values.dot(weights)
        )

        # add synthetic to treatment cluster
        return df[self._get_cluster_column(df) == treatment_cluster].assign(
            synthetic=synthetic
        )

    def fit_predict_synthetic(
        self,
        df: pd.DataFrame,
        pre_experiment_df: pd.DataFrame,
        treatment_cluster: str,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Fit the synthetic control model and predict the results for the treatment cluster.
        Args:
            df: The dataframe containing the data after the intervention.
            pre_experiment_df: The dataframe containing the data before the intervention.
            treatment_cluster: The name of the treatment cluster.
            verbose: If True, print the status of the optimization of weights.

        Returns:
            The dataframe with the synthetic results added to the treatment cluster.
        """
        weights = self._fit(pre_experiment_df=pre_experiment_df, verbose=verbose)

        prediction = self._predict(
            df=df, weights=weights, treatment_cluster=treatment_cluster
        )
        return prediction

    def pvalue_based_on_hypothesis(
        self, ate: np.float64, avg_effects: Dict[str, float]
    ) -> float:
        """
        Returns the p-value of the analysis.
        1. Count how many times the average effect is greater than the real treatment unit
        2. Average it with the number of units. The result is the p-value using Fisher permutation exact test.
        """

        avg_effects = list(avg_effects.values())

        if HypothesisEntries(self.hypothesis) == HypothesisEntries.LESS:
            return np.mean(avg_effects < ate)
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.GREATER:
            return np.mean(avg_effects > ate)
        if HypothesisEntries(self.hypothesis) == HypothesisEntries.TWO_SIDED:
            avg_effects = np.abs(avg_effects)
            return np.mean(avg_effects > ate)

        raise ValueError(f"{self.hypothesis} is not a valid HypothesisEntries")

    def _get_treatment_cluster(self, df: pd.DataFrame) -> str:
        """Returns the first treatment cluster. The current implementation of Synthetic Control only accepts one treatment cluster.
        This will be left inside Synthetic class because it doesn't apply for other analyses
        """
        treatment_df = df[df[self.treatment_col] == 1]
        treatment_cluster = self._get_cluster_column(treatment_df).unique()[0]
        return treatment_cluster

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """
        Returns the p-value of the analysis.
        1. Calculate the average effect after intervention for each unit.
        2. Count how many times the average effect is greater than the real treatment unit
        3. Average it with the number of units. The result is the p-value using Fisher permutation test
        """

        clusters = self._get_cluster_column(df).unique()
        treatment_cluster = self._get_treatment_cluster(df)

        synthetic_donors = {
            cluster: self.analysis_point_estimate(
                treatment_cluster=cluster,
                df=df,
                verbose=verbose,
            )
            for cluster in clusters
        }

        ate = synthetic_donors[treatment_cluster]
        synthetic_donors.pop(treatment_cluster)

        return self.pvalue_based_on_hypothesis(ate=ate, avg_effects=synthetic_donors)

    def analysis_point_estimate(
        self,
        df: pd.DataFrame,
        treatment_cluster: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Calculate the point estimate for the treatment effect for a specified cluster by averaging across the time windows.
        """
        df, pre_experiment_df = self._split_pre_experiment_df(df)

        if treatment_cluster is None:
            treatment_cluster = self._get_treatment_cluster(df)

        df = self.fit_predict_synthetic(
            df, pre_experiment_df, treatment_cluster, verbose=verbose
        )

        df["effect"] = df[self.target_col] - df["synthetic"]
        avg_effect = df["effect"].mean()
        return avg_effect

    def _split_pre_experiment_df(self, df: pd.DataFrame):
        """Split the dataframe into pre-experiment and experiment dataframes"""
        pre_experiment_df = df[(df[self.time_col] <= self.intervention_date)]
        df = df[(df[self.time_col] > self.intervention_date)]
        return df, pre_experiment_df


class DeltaMethodAnalysis(ExperimentAnalysis):
    def __init__(
        self,
        cluster_cols: Optional[List[str]] = None,
        target_col: str = "target",
        scale_col: str = "scale",
        treatment_col: str = "treatment",
        treatment: str = "B",
        covariates: Optional[List[str]] = None,
        ratio_covariates: Optional[List[Tuple[str]]] = None,
        hypothesis: str = "two-sided",
    ):
        """
        Class to run the Delta Method approximation for estimating the treatment effect on a ratio metric (target/scale) under a clustered design.
        The analysis is done on the aggregated data at the cluster level, making computation more efficient.

        Arguments:
            cluster_cols: list of columns to use as clusters. Not available for the CUPED method.
            target_col: name of the column containing the variable to measure (the numerator of the ratio).
            scale_col: name of the column containing the scale variable (the denominator of the ratio).
            treatment_col: name of the column containing the treatment variable.
            treatment: name of the treatment to use as the treated group.
            covariates: list of columns to use as covariates.
            ratio_covariates: list of tuples of columns to use as covariates for ratio metrics. First element is the numerator column, second element is the denominator column.
            hypothesis: one of "two-sided", "less", "greater" indicating the alternative hypothesis.

            Usage:
            ```python
            import pandas as pd

            from cluster_experiments.experiment_analysis import DeltaMethodAnalysis

            df = pd.DataFrame({
                'x': [1, 2, 3, 0, 0, 1] * 2,
                'y': [2, 2, 5, 1, 1, 1] * 2,
                'treatment': ["A"] * 6 + ["B"] * 6,
                'cluster': [1, 2, 3, 1, 2, 3] * 2,
            })

            DeltaMethodAnalysis(
                cluster_cols=['cluster'],
                target_col='x',
                scale_col='y'
            ).get_pvalue(df)
            ```
        """
        self.target_col = target_col
        self.target_metric = target_col
        self.scale_col = scale_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis

        self.covariates = covariates or []
        self.covariates_delta = []
        self.ratio_covariates = ratio_covariates or []

    def _aggregate_to_cluster(
        self, df: pd.DataFrame, strat_treatment: Optional[bool] = True
    ) -> pd.DataFrame:
        """
        Returns an aggreegated dataframe of the target and scale variables at the cluster (and treatment) level.

        Arguments:
            df: dataframe containing the data to analyze
        """
        group_cols = (
            self.cluster_cols + [self.treatment_col]
            if strat_treatment
            else self.cluster_cols
        )
        aggregate_df = df.groupby(by=group_cols, as_index=False).agg(
            {self.target_col: "sum", self.scale_col: "sum"}
        )
        return aggregate_df

    def _get_group_mean_and_variance(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Returns the mean and variance of the ratio metric (target/scale) as estimated by the delta method for a given group (treatment).

        Arguments:
            df: dataframe containing the data to analyze.
        """
        df = self._aggregate_to_cluster(df)
        group_size = len(df)

        if group_size < 1000:
            self.__warn_small_group_size()

        target_mean, scale_mean = df.loc[:, [self.target_col, self.scale_col]].mean()
        target_variance, scale_variance = df.loc[
            :, [self.target_col, self.scale_col]
        ].var()
        target_sum, scale_sum = df.loc[:, [self.target_col, self.scale_col]].sum()

        target_scale_cov = df.loc[:, self.target_col].cov(df.loc[:, self.scale_col])

        group_mean = target_sum / scale_sum
        group_variance = (
            (1 / (scale_mean**2)) * target_variance
            + (target_mean**2) / (scale_mean**4) * scale_variance
            - (2 * target_mean) / (scale_mean**3) * target_scale_cov
        ) / group_size
        return group_mean, group_variance

    def _get_mean_SE(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Returns mean and variance of the ratio metric (target/scale) for a given cluster (i.e. user) computed using the Delta Method.
        Variance reduction is used if covariates are given.
        """

        if self.hypothesis != "two-sided":
            raise ValueError(
                "Delta Method currently only supports two-sided hypothesis"
            )

        is_treatment = df[self.treatment_col] == 1
        treat_mean, treat_var = self._get_group_mean_and_variance(df[is_treatment])
        ctrl_mean, ctrl_var = self._get_group_mean_and_variance(df[~is_treatment])

        mean_diff = treat_mean - ctrl_mean
        SE = np.sqrt(treat_var + ctrl_var)

        return mean_diff, SE

    def analysis_pvalue(self, df: pd.DataFrame) -> float:
        """
        Returns the p-value of the analysis.

        Arguments:
            df: dataframe containing the data to analyze.
        """

        mean_diff, SE = self._get_mean_SE(df)

        z_score = mean_diff / SE
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        return p_value

    def analysis_point_estimate(self, df: pd.DataFrame) -> float:
        """Returns the point estimate of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        mean_diff, _SE = self._get_mean_SE(df)
        return mean_diff

    def analysis_standard_error(self, df: pd.DataFrame) -> float:
        """Returns the standard error of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        _mean_diff, SE = self._get_mean_SE(df)
        return SE

    def __warn_small_group_size(self):
        warnings.warn(
            "Delta Method approximation may not be accurate for small group sizes"
        )
