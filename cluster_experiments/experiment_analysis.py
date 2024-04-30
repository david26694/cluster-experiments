import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from scipy.stats import ttest_ind, ttest_rel

from cluster_experiments.synthetic_control import get_w
from cluster_experiments.utils import HypothesisEntries


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

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """Returns the p-value of the analysis
        Arguments:
            df: dataframe containing the data to analyze
            verbose (Optional): bool, prints the regression summary if True
        """
        results_ols = sm.OLS.from_formula(self.formula, data=df,).fit(
            cov_type=self.cov_type,
            cov_kwds={"groups": self._get_cluster_column(df)},
        )
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
        # Keep in mind that the point estimate of the OLS is the same as the ClusteredOLS
        results_ols = sm.OLS.from_formula(
            self.formula,
            data=df,
        ).fit()
        return results_ols.params[self.treatment_col]


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

    def fit_ols(self, df: pd.DataFrame) -> sm.GEE:
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

    dates = pd.date_range(start_date, end_date, freq="d")

    users = [f"User {i}" for i in range(N)]

    # Create a combination of each date with each user
    combinations = list(product(users, dates))

    target_values = np.random.normal(0, 1, size=len(combinations))

    df = pd.DataFrame(combinations, columns=["user", "date"])
    df["target"] = target_values


    SyntheticControlAnalysis(
        cluster_cols=["user"], time_col="date", transition_date="2022-01-15"
    ).get_pvalue(df)

    ```
    """

    def __init__(
        self,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        cluster_cols: Optional[List[str]] = None,
        hypothesis: str = "two-sided",
        time_col: str = "date",
        intervention_date: str = None,
    ):
        super().__init__(cluster_cols)

        self.target_col = target_col
        self.treatment = treatment
        self.treatment_col = treatment_col
        self.cluster_cols = cluster_cols
        self.hypothesis = hypothesis
        self.time_col = time_col
        self.intervention_date = intervention_date

    # todo  check is that time_col not in cluster_cols

    def get_pvalue(self, df: pd.DataFrame) -> float:
        """Returns the p-value of the analysis

        Arguments:
            df: dataframe containing the data to analyze
        """
        df = df.copy()
        df = self._create_binary_treatment(df)
        self._data_checks(df=df)
        return self.analysis_pvalue(df=df)

    def fit_synthetic(self, pre_experiment_df: pd.DataFrame, verbose: bool) -> list:
        """Returns the weights of each donor"""

        if not any(pre_experiment_df[self.treatment_col] == 1):
            raise ValueError("No treatment unit found in the data.")

        X = (
            pre_experiment_df.query(f"{self.treatment_col}==0")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T
        )

        y = (
            pre_experiment_df.query(f"{self.treatment_col}==1")
            .pivot(index=self.cluster_cols, columns=self.time_col)[self.target_col]
            .T.iloc[:, 0]
        )

        weights = get_w(X, y, verbose)

        return weights

    def fit_predict_synthetic(
        self,
        pre_experiment_df: pd.DataFrame,
        df: pd.DataFrame,
        treatment_cluster,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        This method adds a column with the synthetic results and filter only the treatment unit.

        First, it calculates the weights of each donor in the control group using the `fit_synthetic` method.
        It then uses these weights to create a synthetic control group that closely matches the treatment unit before the intervention.
        The synthetic control group is added to the treatment unit in the dataframe.

        Args:
            pre_experiment_df (pd.DataFrame): The dataframe containing the data before the intervention.
            df (pd.DataFrame): The dataframe containing the data after the intervention.
            treatment_cluster (str): The name of the treatment cluster.

        Returns:
            pd.DataFrame: The dataframe with the synthetic results added to the treatment cluster.
        """
        weights = self.fit_synthetic(
            pre_experiment_df=pre_experiment_df, verbose=verbose
        )

        synthetic = (
            df[self._get_cluster_column(df) != treatment_cluster]
            .pivot(index=self.time_col, columns=self.cluster_cols)[self.target_col]
            .values.dot(weights)
        )

        return df[self._get_cluster_column(df) == treatment_cluster].assign(
            synthetic=synthetic
        )  # add synthetic to treatment cluster

    def pvalue_based_on_hypothesis(self, ate: float, avg_effects: dict) -> float:
        """
        Returns the p-value of the analysis.
        1. calculate the average effect after intervention for each unit.
        2. count how many times the average effect is greater than the real treatment unit
        3. Divide by the number of units. The result is the p-value using Fisher permutation test.
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
        """Returns the treatment cluster"""
        treatment_df = df[df[self.treatment_col] == 1]

        treatment_cluster = self._get_cluster_column(treatment_df).unique()[0]
        return treatment_cluster

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        """
        Calculate the p-value using the treatment effect calculated from analysis_point_estimate
        for the actual treatment cluster and comparing it with placebo effects from other clusters.
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
        Calculate the point estimate for the treatment effect for a specified cluster.
        """
        pre_experiment_df = df.query(f"{self.time_col} < '{self.intervention_date}'")
        df = df.query(f"{self.time_col} >= '{self.intervention_date}'")

        if not treatment_cluster:
            treatment_cluster = self._get_treatment_cluster(df)

        df = self.fit_predict_synthetic(
            pre_experiment_df, df, treatment_cluster, verbose=verbose
        )

        df["effect"] = df[self.target_col] - df["synthetic"]
        avg_effect = df["effect"].mean()
        return avg_effect

    @classmethod
    def from_config(cls, config):
        """Creates an SyntheticControlAnalysis object from a PowerConfig object"""
        return cls(
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            hypothesis=config.hypothesis,
            cluster_cols=config.cluster_cols,
        )
