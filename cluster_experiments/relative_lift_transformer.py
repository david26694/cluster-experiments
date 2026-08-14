from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.regression.linear_model import RegressionResultsWrapper


class RegressionResultsProtocol(Protocol):
    @property
    def params(self) -> Dict[str, float]: ...

    @property
    def bse(self) -> Dict[str, float]: ...

    @property
    def pvalues(self) -> Dict[str, float]: ...

    def conf_int(self, alpha: float) -> Dict[str, Tuple[float, float]]: ...

    def summary(self): ...


class BaseLiftTransformer:
    """
    Base class for relative lift transformers.

    Holds the results of a relative-lift computation and exposes them through the
    ``RegressionResultsProtocol``-compatible interface (``params``, ``bse``,
    ``pvalues``, ``conf_int``, ``summary``), so a relative fit can be dropped into
    code paths that expect a statsmodels results object.

    Alongside the point estimate and its standard error, subclasses record the two
    coefficients describing how that standard error varies with the effect size
    (see :class:`~cluster_experiments.experiment_analysis.StandardErrorCurve`).
    Power analysis reads them via :meth:`standard_error_curve`.

    Note there is deliberately no ``fit`` on the base class: the subclasses take
    entirely different inputs (a fitted OLS model versus delta-method group
    statistics). What they share is the shape of the *results*, not the fitting.
    """

    def __init__(self, treatment_col: str):
        self.treatment_col = treatment_col
        self._relative_lift_value: Optional[float] = None
        self._se_relative_lift: Optional[float] = None
        # Coefficients of SE(m)**2 = se_null**2 + effect_var*m**2 - 2*effect_cov*m
        self._se_null: Optional[float] = None
        self._effect_var: Optional[float] = None
        self._effect_cov: Optional[float] = None

    def _set_results(
        self,
        relative_lift: float,
        se_relative_lift: float,
        se_null: float,
        effect_var: float,
        effect_cov: float,
    ) -> None:
        """Records everything a subclass' ``fit`` computes."""
        self._relative_lift_value = relative_lift
        self._se_relative_lift = se_relative_lift
        self._se_null = se_null
        self._effect_var = effect_var
        self._effect_cov = effect_cov

    def standard_error_curve(self):
        """
        Returns the relative-lift standard error as a function of the true effect.

        Note ``std_error`` on the returned curve is the standard error under the
        null, which differs from ``bse`` — the latter is evaluated at the observed
        lift, which is what inference needs.
        """
        # Imported here to avoid a circular import at module load time.
        from cluster_experiments.experiment_analysis import StandardErrorCurve

        if self._se_null is None:
            raise ValueError("fit must be called before standard_error_curve")
        return StandardErrorCurve(
            std_error=self._se_null,
            effect_var=self._effect_var,
            effect_cov=self._effect_cov,
        )

    @property
    def params(self):
        return {self.treatment_col: self._relative_lift_value}

    @property
    def bse(self):
        return {self.treatment_col: self._se_relative_lift}

    @property
    def pvalues(self):
        z_score = self._relative_lift_value / self._se_relative_lift
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        return {self.treatment_col: p_value}

    def conf_int(self, alpha: float):
        # 1. Critical value
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # 2. Confidence interval
        lower_bound = self._relative_lift_value - z_crit * self._se_relative_lift
        upper_bound = self._relative_lift_value + z_crit * self._se_relative_lift

        return pd.DataFrame(
            [[lower_bound, upper_bound]],
            index=[self.treatment_col],
            columns=[0, 1],
        )

    def summary(self, alpha: float = 0.05):
        return {
            "percent_lift": self._relative_lift_value,
            "_se_relative_lift": self._se_relative_lift,
            "pvalue": self.pvalues[self.treatment_col],
            "conf_int": self.conf_int(alpha).loc[self.treatment_col],
        }


class LiftRegressionTransformer(BaseLiftTransformer):
    def fit(
        self, ols: RegressionResultsWrapper, df: pd.DataFrame, covariate_cols: List[str]
    ) -> None:
        """
        Stores values of relative lift and relative standard error.
        1. Compatible with covariates
        2. Using delta method,

        Let the regression model be:

            Y_i = intercept + treatment_i * tau + X_i * beta + epsilon_i

        where:
        - tau: treatment effect coefficient
        - beta: covariate coefficients
        - X_i: covariates
        - epsilon_i: residual

        The **adjusted control mean** is:

            adjusted_control_mean = intercept + mean(X_control) @ beta

        The **percent lift** is:

            percent_lift = tau / adjusted_control_mean

        The **variance of percent lift** via the delta method is:

            Var(percent_lift) = (Var(tau) / adjusted_control_mean^2)
                                + (tau^2 / adjusted_control_mean^4) * Var(adjusted_control_mean)
                                - 2 * (tau / adjusted_control_mean^3) * Cov(tau, adjusted_control_mean)

        If covariates are centered (or no covariates), adjusted_control_mean = intercept

        """
        coefficients = ols.params
        covariance_matrix = ols.cov_params()

        intercept_value = coefficients["Intercept"]
        treatment_effect = coefficients[self.treatment_col]
        covariate_effects = coefficients[covariate_cols].values

        # 1. Control group covariate mean
        control_covariates = df.loc[df[self.treatment_col] == 0, covariate_cols].values
        control_covariates_mean = control_covariates.mean(axis=0)

        # 2. Regression-adjusted control group mean
        adjusted_control_mean = (
            intercept_value + control_covariates_mean @ covariate_effects
        )

        # 3. Variance of adjusted control mean
        var_intercept = covariance_matrix.loc["Intercept", "Intercept"]
        cov_intercept_covariates = covariance_matrix.loc[
            "Intercept", covariate_cols
        ].values
        cov_covariates = covariance_matrix.loc[covariate_cols, covariate_cols].values
        var_adjusted_control_mean = (
            var_intercept
            + control_covariates_mean @ cov_covariates @ control_covariates_mean
            + 2 * cov_intercept_covariates @ control_covariates_mean
        )

        # 4. Covariance between treatment effect and adjusted control mean
        cov_treatment_intercept = covariance_matrix.loc[self.treatment_col, "Intercept"]
        cov_treatment_covariates = covariance_matrix.loc[
            self.treatment_col, covariate_cols
        ].values
        cov_treatment_control_mean = (
            cov_treatment_intercept + cov_treatment_covariates @ control_covariates_mean
        )

        # 5. Percent lift
        _relative_lift_value = treatment_effect / adjusted_control_mean

        # 6. Delta-method variance for percent lift.
        #
        # Written in terms of the lift m = tau / mu_c, this is the quadratic
        #     Var(m) = se_null**2 + effect_var * m**2 - 2 * effect_cov * m
        # which is what power analysis needs to know how the standard error grows
        # with the effect size. Evaluating it at the observed lift gives the
        # standard error used for inference.
        se_null_squared = (
            covariance_matrix.loc[self.treatment_col, self.treatment_col]
            / adjusted_control_mean**2
        )
        effect_var = var_adjusted_control_mean / adjusted_control_mean**2
        effect_cov = cov_treatment_control_mean / adjusted_control_mean**2

        var_percent_lift = (
            se_null_squared
            + effect_var * _relative_lift_value**2
            - 2 * effect_cov * _relative_lift_value
        )
        _se_relative_lift = np.sqrt(var_percent_lift)

        self._set_results(
            relative_lift=_relative_lift_value,
            se_relative_lift=_se_relative_lift,
            se_null=float(np.sqrt(se_null_squared)),
            effect_var=float(effect_var),
            effect_cov=float(effect_cov),
        )


class DeltaMethodLiftTransformer(BaseLiftTransformer):
    """
    Delta-method relative lift for ratio metrics (cluster-level target/scale).

    Mirrors the ``LiftRegressionTransformer`` instance API: call :meth:`fit` with
    the statistics produced by ``DeltaMethodAnalysis._get_group_statistics``, then
    read `params`, `bse`, `pvalues`, and `conf_int` just like any
    ``RegressionResultsProtocol``-compatible object, or
    :meth:`standard_error_curve` for power analysis.

    The static helper :meth:`lift_and_se` remains available for direct use.
    """

    def fit(
        self,
        mean_diff: float,
        std_error: float,
        ctrl_mean: float,
        ctrl_var: float,
    ) -> None:
        """
        Compute and store the relative lift and its SE via the outer delta method.

        Parameters
        ----------
        mean_diff
            Absolute treatment effect on the ratio metric (treat_mean - ctrl_mean).
        std_error
            Standard error of mean_diff = sqrt(treat_var + ctrl_var).
        ctrl_mean
            Control arm ratio mean.
        ctrl_var
            Variance of the control arm ratio mean.
        """
        relative_lift, se = self.lift_and_se(
            mean_diff, std_error**2, ctrl_mean, ctrl_var
        )
        # The arms are independent, so Cov(mean_diff, ctrl_mean) = -ctrl_var and
        # the covariance coefficient is minus the variance coefficient. That is
        # the special case for which SE(m)**2 collapses to
        # se2_t + se2_c * (1 + m)**2.
        effect_var = ctrl_var / ctrl_mean**2
        self._set_results(
            relative_lift=relative_lift,
            se_relative_lift=se,
            se_null=float(std_error / abs(ctrl_mean)),
            effect_var=float(effect_var),
            effect_cov=float(-effect_var),
        )

    @staticmethod
    def lift_and_se(
        mean_diff: float,
        var_abs: float,
        ctrl_mean: float,
        ctrl_var: float,
    ) -> Tuple[float, float]:
        """
        Relative lift (mean_diff / ctrl_mean) and SE via the outer delta method.

        Parameters
        ----------
        mean_diff
            Absolute treatment effect on the ratio metric.
        var_abs
            Var(mean_diff) = treat_var + ctrl_var.
        ctrl_mean
            Control arm ratio mean.
        ctrl_var
            Variance of control arm ratio mean.
        """
        if ctrl_mean == 0:
            raise ValueError("ctrl_mean must be non-zero for relative lift.")
        relative_lift = mean_diff / ctrl_mean
        var_relative = (
            var_abs / (ctrl_mean**2)
            + (mean_diff**2) * ctrl_var / (ctrl_mean**4)
            + 2 * mean_diff * ctrl_var / (ctrl_mean**3)
        )
        return relative_lift, float(np.sqrt(var_relative))
