"""Standard error of an estimate as a function of the effect size.

Kept in its own module so both the analyses and the relative-lift transformers can
import it without a cycle.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StandardErrorCurve:
    """
    Standard error of an estimate as a function of the true effect ``m``::

        SE(m)**2 = std_error**2 + effect_var * m**2 - 2 * effect_cov * m

    For an **absolute** effect the standard error does not depend on the effect
    size, so ``effect_var = effect_cov = 0`` and ``standard_error_at`` is constant.

    For a **relative** effect it does depend on the effect size, because the
    denominator of the lift is itself estimated. Writing the lift as a function
    of the numerator and the (estimated) baseline and applying the delta method
    gives the quadratic above, for both estimators that support relative effects:

    ==========================  ==========================  ==================
    quantity                    :class:`OLSAnalysis`        delta method
    ==========================  ==========================  ==================
    ``std_error**2``            ``Var(tau) / mu_c**2``      ``(treat_var + ctrl_var) / r_c**2``
    ``effect_var``              ``Var(mu_c) / mu_c**2``     ``ctrl_var / r_c**2``
    ``effect_cov``              ``Cov(tau, mu_c) / mu_c**2``  ``-ctrl_var / r_c**2``
    ==========================  ==========================  ==================

    where ``tau`` is the regression treatment coefficient, ``mu_c`` the
    (covariate-adjusted) control mean, and ``r_c`` the control ratio mean. The
    delta-method column is the special case ``effect_cov = -effect_var``, for
    which the formula collapses to ``se2_t + se2_c * (1 + m)**2``.

    :class:`~cluster_experiments.power_analysis.NormalPowerAnalysis` derives both
    power and the MDE from this single object, which is what makes them exact
    inverses of each other.

    Attributes:
        std_error: Standard error under the null, i.e. ``SE(0)``. This is the
            quantity power and MDE calculations need. Note that it is *not* the
            standard error reported by inference methods for a relative effect,
            which is evaluated at the observed effect instead.
        effect_var: Variance of the baseline, normalised by the squared baseline.
            Zero for absolute effects.
        effect_cov: Covariance between the numerator and the baseline, normalised
            by the squared baseline. Zero for absolute effects.

    Usage:
    ```python
    from cluster_experiments import StandardErrorCurve

    # An absolute effect: the standard error does not vary with the effect size.
    absolute = StandardErrorCurve(std_error=0.05)
    print(absolute.is_effect_dependent)
    print(absolute.standard_error_at(0.0) == absolute.standard_error_at(0.5))

    # A relative effect on a ratio metric, where the two arms are independent so
    # effect_cov is minus effect_var.
    relative = StandardErrorCurve(std_error=0.05, effect_var=0.01, effect_cov=-0.01)
    print(relative.is_effect_dependent)
    print(relative.standard_error_at(0.0))
    print(relative.standard_error_at(0.5) > relative.standard_error_at(0.0))
    ```
    """

    std_error: float
    effect_var: float = 0.0
    effect_cov: float = 0.0

    @property
    def is_effect_dependent(self) -> bool:
        """
        True when the standard error varies with the effect size.
        """
        return self.effect_var != 0.0 or self.effect_cov != 0.0

    def standard_error_at(self, effect: float) -> float:
        """
        Standard error of the estimate when the true effect is ``effect``.

        Arguments:
            effect: the true effect size, on the same scale as the estimate.

        Usage:
        ```python
        from cluster_experiments import StandardErrorCurve

        curve = StandardErrorCurve(std_error=0.05, effect_var=0.01, effect_cov=-0.01)
        # SE(0) is the standard error under the null
        print(round(curve.standard_error_at(0.0), 6))
        # and it grows with the effect, because the baseline is itself estimated
        print(round(curve.standard_error_at(0.2), 6))
        ```
        """
        if not self.is_effect_dependent:
            return self.std_error
        variance = (
            self.std_error**2
            + self.effect_var * effect**2
            - 2 * self.effect_cov * effect
        )
        # A variance this far from the null is outside the range the delta-method
        # expansion describes; clamp rather than return a nan.
        return float(np.sqrt(max(variance, 0.0)))
