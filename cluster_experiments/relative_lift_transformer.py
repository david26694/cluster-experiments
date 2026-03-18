"""
Relative (percent) lift: OLS path via `LiftRegressionTransformer`; ratio metrics
(target/scale) via `DeltaMethodLiftTransformer` (outer delta + quadratic relative MDE).
"""

from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
from statsmodels.regression.linear_model import RegressionResultsWrapper


class DeltaMethodLiftTransformer:
    """
    Delta-method relative lift and MDE for ratio metrics (cluster-level target/scale).

    Static API only; no regression object required.
    """

    @staticmethod
    def lift_and_se(
        mean_diff: float,
        var_abs: float,
        ctrl_mean: float,
        ctrl_var: float,
    ) -> Tuple[float, float]:
        """
        Relative lift (ATE / control ratio mean) and SE (outer delta method).

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

    @staticmethod
    def relative_mde(
        alpha: float,
        power: float,
        ctrl_mean: float,
        ctrl_var: float,
        treat_var: float,
    ) -> float:
        """
        Minimum detectable relative lift (two-sided, double-delta quadratic).
        """
        if ctrl_mean == 0:
            raise ValueError("ctrl_mean must be non-zero for relative MDE.")
        r_c = ctrl_mean
        se2_c = ctrl_var / (r_c**2)
        se2_t = treat_var / (r_c**2)

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        v0 = se2_t + se2_c
        c = z_alpha * np.sqrt(v0)

        a = 1 - (z_beta**2) * se2_c
        b = -2 * (c + (z_beta**2) * se2_c)
        c_term = c**2 - (z_beta**2) * v0

        discriminant = b**2 - 4 * a * c_term
        if discriminant < 0 or a == 0:
            raise ValueError(
                "DeltaMethodLiftTransformer.relative_mde: invalid power equation "
                "(discriminant or A); check inputs or use more clusters."
            )
        m = (-b + np.sqrt(discriminant)) / (2 * a)
        return float(m)


def ratio_relative_lift_and_se(
    mean_diff: float,
    var_abs: float,
    ctrl_mean: float,
    ctrl_var: float,
) -> Tuple[float, float]:
    """Backward-compatible wrapper for :meth:`DeltaMethodLiftTransformer.lift_and_se`."""
    return DeltaMethodLiftTransformer.lift_and_se(
        mean_diff, var_abs, ctrl_mean, ctrl_var
    )


def relative_ratio_mde(
    alpha: float,
    power: float,
    ctrl_mean: float,
    ctrl_var: float,
    treat_var: float,
) -> float:
    """Backward-compatible wrapper for :meth:`DeltaMethodLiftTransformer.relative_mde`."""
    return DeltaMethodLiftTransformer.relative_mde(
        alpha, power, ctrl_mean, ctrl_var, treat_var
    )


class RegressionResultsProtocol(Protocol):
    @property
    def params(self) -> Dict[str, float]: ...

    @property
    def bse(self) -> Dict[str, float]: ...

    @property
    def pvalues(self) -> Dict[str, float]: ...

    def conf_int(self, alpha: float) -> Dict[str, Tuple[float, float]]: ...

    def summary(self): ...


class LiftRegressionTransformer:
    def __init__(self, treatment_col: str):
        self.treatment_col = treatment_col
        self._relative_lift_value: Optional[float] = None
        self._se_relative_lift: Optional[float] = None

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

        # 5. Delta-method variance for percent lift
        var_percent_lift = (
            covariance_matrix.loc[self.treatment_col, self.treatment_col]
            / adjusted_control_mean**2
            + (treatment_effect**2 / adjusted_control_mean**4)
            * var_adjusted_control_mean
            - 2
            * (treatment_effect / adjusted_control_mean**3)
            * cov_treatment_control_mean
        )
        _se_relative_lift = np.sqrt(var_percent_lift)

        # 6. Percent lift
        _relative_lift_value = treatment_effect / adjusted_control_mean

        self._relative_lift_value = _relative_lift_value
        self._se_relative_lift = _se_relative_lift

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

    def summary(self):
        return {
            "percent_lift": self._relative_lift_value,
            "_se_relative_lift": self._se_relative_lift,
            "pvalue": self.pvalues[self.treatment_col],
            "conf_int": self.conf_int(0.05).loc[self.treatment_col],
        }
