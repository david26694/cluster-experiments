"""Generate plot images for plot_experiment_results across edge cases.

Run from repo root:

    uv run --extra dev python examples/plotting_outputs/generate_plots.py

All PNGs are written next to this file.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cluster_experiments import plot_experiment_results  # noqa: E402

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class FakeResults:
    """Minimal AnalysisPlanResults-compatible stub."""

    ate: List[float]
    ate_ci_lower: List[float]
    ate_ci_upper: List[float]
    p_value: List[float]
    control_variant_mean: List[float] = field(default_factory=list)
    treatment_variant_mean: List[float] = field(default_factory=list)
    treatment_variant_name: List[str] = field(default_factory=list)
    control_variant_name: List[str] = field(default_factory=list)
    alpha: List[float] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.ate)
        if not self.control_variant_mean:
            self.control_variant_mean = [0.10] * n
        if not self.treatment_variant_mean:
            self.treatment_variant_mean = [
                c + a for c, a in zip(self.control_variant_mean, self.ate)
            ]
        if not self.treatment_variant_name:
            self.treatment_variant_name = [f"variant_{i + 1}" for i in range(n)]
        if not self.control_variant_name:
            self.control_variant_name = ["control"] * n
        if not self.alpha:
            self.alpha = [0.05] * n


def save(ax, name: str, title: Optional[str] = None) -> str:
    fig = ax.figure
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def scenario_2_variant_relative():
    result = FakeResults(
        ate=[0.03],
        ate_ci_lower=[0.012],
        ate_ci_upper=[0.048],
        p_value=[0.01],
        control_variant_mean=[0.10],
        treatment_variant_name=["treatment"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(
        ax, "01_2variant_relative_significant.png", "A/B - relative lift (significant)"
    )


def scenario_2_variant_absolute():
    result = FakeResults(
        ate=[3.0],
        ate_ci_lower=[1.2],
        ate_ci_upper=[4.8],
        p_value=[0.01],
        control_variant_mean=[100.0],
        treatment_variant_mean=[103.0],
        treatment_variant_name=["treatment"],
    )
    ax = plot_experiment_results(result, metric_type="absolute")
    return save(ax, "02_2variant_absolute_significant.png", "A/B - absolute lift")


def scenario_3_variant_mixed():
    result = FakeResults(
        ate=[0.025, -0.018, 0.004],
        ate_ci_lower=[0.010, -0.032, -0.011],
        ate_ci_upper=[0.040, -0.004, 0.019],
        p_value=[0.01, 0.02, 0.55],
        control_variant_mean=[0.10, 0.10, 0.10],
        treatment_variant_name=["treatment_A", "treatment_B", "treatment_C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(
        ax, "03_3variant_mixed_signs.png", "A/B/C - mixed (positive, negative, null)"
    )


def scenario_4_variant_relative():
    result = FakeResults(
        ate=[0.012, 0.028, -0.006, 0.045],
        ate_ci_lower=[0.002, 0.016, -0.017, 0.030],
        ate_ci_upper=[0.022, 0.040, 0.005, 0.060],
        p_value=[0.04, 0.001, 0.35, 0.0001],
        control_variant_mean=[0.10] * 4,
        treatment_variant_name=["variant_A", "variant_B", "variant_C", "variant_D"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "04_4variant_relative.png", "A/B/C/D - relative lift")


def scenario_4_variant_absolute():
    result = FakeResults(
        ate=[1.2, 2.8, -0.6, 4.5],
        ate_ci_lower=[0.2, 1.6, -1.7, 3.0],
        ate_ci_upper=[2.2, 4.0, 0.5, 6.0],
        p_value=[0.04, 0.001, 0.35, 0.0001],
        control_variant_mean=[100.0] * 4,
        treatment_variant_mean=[101.2, 102.8, 99.4, 104.5],
        treatment_variant_name=["variant_A", "variant_B", "variant_C", "variant_D"],
    )
    ax = plot_experiment_results(result, metric_type="absolute")
    return save(ax, "05_4variant_absolute.png", "A/B/C/D - absolute lift")


def scenario_5_variant_many():
    result = FakeResults(
        ate=[0.008, 0.016, 0.024, 0.032, 0.040],
        ate_ci_lower=[-0.004, 0.004, 0.012, 0.020, 0.028],
        ate_ci_upper=[0.020, 0.028, 0.036, 0.044, 0.052],
        p_value=[0.20, 0.03, 0.001, 0.0005, 0.0001],
        control_variant_mean=[0.50] * 5,
        treatment_variant_name=[f"dose_{d}" for d in (10, 20, 40, 80, 160)],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "06_5variant_dose_response.png", "5-arm dose response (relative)")


def scenario_all_significant_positive():
    result = FakeResults(
        ate=[0.02, 0.03, 0.04],
        ate_ci_lower=[0.01, 0.02, 0.03],
        ate_ci_upper=[0.03, 0.04, 0.05],
        p_value=[0.001, 0.001, 0.001],
        control_variant_mean=[0.10] * 3,
        treatment_variant_name=["A", "B", "C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "07_all_positive_significant.png", "All positive & significant")


def scenario_all_negative_significant():
    result = FakeResults(
        ate=[-0.02, -0.03, -0.04],
        ate_ci_lower=[-0.03, -0.04, -0.05],
        ate_ci_upper=[-0.01, -0.02, -0.03],
        p_value=[0.001, 0.001, 0.001],
        control_variant_mean=[0.10] * 3,
        treatment_variant_name=["A", "B", "C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "08_all_negative_significant.png", "All negative & significant")


def scenario_none_significant():
    result = FakeResults(
        ate=[0.005, -0.003, 0.001],
        ate_ci_lower=[-0.015, -0.020, -0.017],
        ate_ci_upper=[0.025, 0.014, 0.019],
        p_value=[0.40, 0.65, 0.88],
        control_variant_mean=[0.10] * 3,
        treatment_variant_name=["A", "B", "C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "09_none_significant.png", "None significant")


def scenario_zero_control_mean():
    result = FakeResults(
        ate=[0.5, -0.3],
        ate_ci_lower=[0.1, -0.7],
        ate_ci_upper=[0.9, 0.1],
        p_value=[0.02, 0.10],
        control_variant_mean=[0.0, 0.0],
        treatment_variant_mean=[0.5, -0.3],
        treatment_variant_name=["A", "B"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(
        ax,
        "10_zero_control_mean_infinite_lift.png",
        "Zero control mean -> infinite relative lift",
    )


def scenario_nan_values():
    nan = math.nan
    result = FakeResults(
        ate=[0.02, nan, 0.015],
        ate_ci_lower=[0.01, nan, 0.002],
        ate_ci_upper=[0.03, nan, 0.028],
        p_value=[0.01, nan, 0.03],
        control_variant_mean=[0.10, 0.10, 0.10],
        treatment_variant_name=["A", "B", "C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "11_nan_values.png", "Missing/NaN row rendered as gray x")


def scenario_missing_ci():
    nan = math.nan
    result = FakeResults(
        ate=[0.02, 0.03],
        ate_ci_lower=[nan, 0.015],
        ate_ci_upper=[nan, 0.045],
        p_value=[0.01, 0.02],
        control_variant_mean=[0.10, 0.10],
        treatment_variant_name=["A_no_ci", "B"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "12_missing_ci.png", "Missing CI bounds")


def scenario_missing_p_values():
    nan = math.nan
    result = FakeResults(
        ate=[0.02, -0.015],
        ate_ci_lower=[0.01, -0.025],
        ate_ci_upper=[0.03, -0.005],
        p_value=[nan, nan],
        control_variant_mean=[0.10, 0.10],
        treatment_variant_name=["A", "B"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "13_missing_p_values.png", "Missing p-values -> gray")


def scenario_large_effects():
    result = FakeResults(
        ate=[0.35, -0.28, 0.60],
        ate_ci_lower=[0.20, -0.45, 0.40],
        ate_ci_upper=[0.50, -0.10, 0.80],
        p_value=[0.001, 0.001, 0.0001],
        control_variant_mean=[1.0] * 3,
        treatment_variant_name=["A", "B", "C"],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "14_large_effects.png", "Large effects (broad CI range)")


def scenario_custom_ax_subplots():
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    rel = FakeResults(
        ate=[0.02, -0.01, 0.035],
        ate_ci_lower=[0.01, -0.02, 0.020],
        ate_ci_upper=[0.03, 0.00, 0.050],
        p_value=[0.01, 0.20, 0.001],
        control_variant_mean=[0.10] * 3,
        treatment_variant_name=["A", "B", "C"],
    )
    abs_ = FakeResults(
        ate=[2.0, -1.0, 3.5],
        ate_ci_lower=[1.0, -2.0, 2.0],
        ate_ci_upper=[3.0, 0.0, 5.0],
        p_value=[0.01, 0.20, 0.001],
        control_variant_mean=[100.0] * 3,
        treatment_variant_mean=[102.0, 99.0, 103.5],
        treatment_variant_name=["A", "B", "C"],
    )
    plot_experiment_results(rel, metric_type="relative", ax=axes[0], title="Relative")
    plot_experiment_results(abs_, metric_type="absolute", ax=axes[1], title="Absolute")
    path = os.path.join(OUTPUT_DIR, "15_custom_ax_stacked.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def scenario_custom_alpha_strict():
    result = FakeResults(
        ate=[0.010, 0.015, 0.020],
        ate_ci_lower=[0.002, 0.006, 0.011],
        ate_ci_upper=[0.018, 0.024, 0.029],
        p_value=[0.04, 0.02, 0.001],
        control_variant_mean=[0.10] * 3,
        treatment_variant_name=["A", "B", "C"],
        alpha=[0.01, 0.01, 0.01],
    )
    ax = plot_experiment_results(result, metric_type="relative")
    return save(ax, "16_strict_alpha_001.png", "alpha=0.01 (A & B become gray)")


def main():
    scenarios = [
        scenario_2_variant_relative,
        scenario_2_variant_absolute,
        scenario_3_variant_mixed,
        scenario_4_variant_relative,
        scenario_4_variant_absolute,
        scenario_5_variant_many,
        scenario_all_significant_positive,
        scenario_all_negative_significant,
        scenario_none_significant,
        scenario_zero_control_mean,
        scenario_nan_values,
        scenario_missing_ci,
        scenario_missing_p_values,
        scenario_large_effects,
        scenario_custom_ax_subplots,
        scenario_custom_alpha_strict,
    ]
    paths = []
    for scenario in scenarios:
        path = scenario()
        paths.append(path)
        print(f"wrote {os.path.relpath(path)}")
    print(f"\n{len(paths)} plots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
