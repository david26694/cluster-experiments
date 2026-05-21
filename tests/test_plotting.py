import builtins
from dataclasses import dataclass, field

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from cluster_experiments.plotting import plot_experiment_results


@dataclass
class FakeResults:
    control_variant_mean: list = field(default_factory=list)
    treatment_variant_mean: list = field(default_factory=list)
    ate: list = field(default_factory=list)
    ate_ci_lower: list = field(default_factory=list)
    ate_ci_upper: list = field(default_factory=list)
    p_value: list = field(default_factory=list)
    alpha: list = field(default_factory=list)
    control_variant_name: list = field(default_factory=list)
    treatment_variant_name: list = field(default_factory=list)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def result(**kwargs):
    defaults = {
        "control_variant_mean": [0.10],
        "treatment_variant_mean": [0.11],
        "ate": [0.01],
        "ate_ci_lower": [0.002],
        "ate_ci_upper": [0.018],
        "p_value": [0.01],
        "alpha": [0.05],
        "control_variant_name": ["A"],
        "treatment_variant_name": ["B"],
    }
    defaults.update(kwargs)
    return FakeResults(**defaults)


def marker_lines(ax):
    return [line for line in ax.lines if line.get_marker() in {"o", "x"}]


def marker_colors(ax):
    return [line.get_color() for line in marker_lines(ax)]


def marker_kinds(ax):
    return [line.get_marker() for line in marker_lines(ax)]


def ci_rectangles(ax):
    return [
        patch
        for patch in ax.patches
        if isinstance(patch, Rectangle)
        and patch.get_facecolor()[:3] != (0.0, 0.0, 0.0)
        and patch.get_width() != 0
    ]


def circles(ax):
    return [patch for patch in ax.patches if isinstance(patch, Circle)]


def texts(ax):
    return [text.get_text() for text in ax.texts]


def test_single_treatment_relative_renders_scoreboard():
    ax = plot_experiment_results(result(), metric_type="relative", title="Result")

    assert ax.get_title() == "Result"
    assert len(circles(ax)) == 2
    assert "B vs A" not in texts(ax)
    assert any(text.endswith("%") and "+10.00" in text for text in texts(ax))
    assert "10.00" in texts(ax)
    assert "11.00" in texts(ax)


def test_single_treatment_absolute_uses_raw_values_and_no_percent_suffix():
    ax = plot_experiment_results(
        result(
            control_variant_mean=[10.0],
            treatment_variant_mean=[11.0],
            ate=[1.0],
            ate_ci_lower=[0.2],
            ate_ci_upper=[1.8],
        ),
        metric_type="absolute",
    )

    assert any("+1.00" in text and "%" not in text for text in texts(ax))
    assert "10.00" in texts(ax)
    assert "11.00" in texts(ax)


def test_multi_variant_relative_renders_one_row_per_treatment():
    ax = plot_experiment_results(
        result(
            control_variant_mean=[0.10, 0.20, 0.50],
            treatment_variant_mean=[0.11, 0.19, 0.55],
            ate=[0.01, -0.01, 0.05],
            ate_ci_lower=[0.005, -0.02, 0.01],
            ate_ci_upper=[0.015, 0.0, 0.09],
            p_value=[0.01, 0.02, 0.2],
            control_variant_name=["A", "A", "A"],
            treatment_variant_name=["B", "C", "D"],
        ),
        metric_type="relative",
    )

    assert len(circles(ax)) == 6
    assert len(marker_lines(ax)) == 3
    text_blob = "\n".join(texts(ax))
    assert "+10.00%" in text_blob
    assert "-5.00%" in text_blob
    assert "+10.00%" in text_blob


def test_multi_variant_absolute_renders_one_row_per_treatment():
    ax = plot_experiment_results(
        result(
            control_variant_mean=[10.0, 10.0, 10.0],
            treatment_variant_mean=[11.0, 8.0, 13.0],
            ate=[1.0, -2.0, 3.0],
            ate_ci_lower=[0.0, -3.0, 1.0],
            ate_ci_upper=[2.0, -1.0, 5.0],
            p_value=[0.01, 0.01, 0.001],
            treatment_variant_name=["B", "C", "D"],
            control_variant_name=["A", "A", "A"],
        ),
        metric_type="absolute",
    )

    assert len(marker_lines(ax)) == 3
    text_blob = "\n".join(texts(ax))
    assert "+1.00" in text_blob
    assert "-2.00" in text_blob
    assert "+3.00" in text_blob


def test_zero_control_mean_relative_renders_infinite_lift_without_crashing():
    ax = plot_experiment_results(
        result(control_variant_mean=[0.0], ate=[1.0]),
        metric_type="relative",
    )

    assert marker_kinds(ax) == ["x"]
    assert any("+inf" in text for text in texts(ax))


def test_missing_ci_does_not_crash_and_skips_ci_bar():
    ax = plot_experiment_results(
        result(
            control_variant_mean=[10.0],
            treatment_variant_mean=[11.0],
            ate=[1.0],
            ate_ci_lower=[],
            ate_ci_upper=[],
        ),
        metric_type="absolute",
    )

    assert ci_rectangles(ax) == []
    assert marker_kinds(ax) == ["o"]


def test_missing_p_values_render_lift_in_gray():
    ax = plot_experiment_results(result(p_value=[]), metric_type="absolute")
    gray = "#7F8C8D"
    assert any(text.get_color() == gray and "+" in text.get_text() for text in ax.texts)


def test_negative_significant_lift_is_red_with_down_icon():
    ax = plot_experiment_results(
        result(
            ate=[-0.01],
            ate_ci_lower=[-0.02],
            ate_ci_upper=[-0.002],
            p_value=[0.01],
        ),
        metric_type="relative",
    )
    red = "#E74C3C"
    assert any(
        text.get_color() == red and "\u25bc" in text.get_text() for text in ax.texts
    )


def test_positive_significant_lift_is_green_with_up_icon():
    ax = plot_experiment_results(result(), metric_type="relative")
    green = "#2ECC71"
    assert any(
        text.get_color() == green and "\u25b2" in text.get_text() for text in ax.texts
    )


def test_non_significant_lift_is_gray():
    ax = plot_experiment_results(result(p_value=[0.8]), metric_type="relative")
    gray = "#7F8C8D"
    assert any(text.get_color() == gray and "+" in text.get_text() for text in ax.texts)


def test_nan_values_do_not_crash():
    nan = float("nan")
    ax = plot_experiment_results(
        result(ate=[nan], ate_ci_lower=[nan], ate_ci_upper=[nan])
    )
    assert marker_kinds(ax) == ["x"]
    assert any("nan" in text for text in texts(ax))


def test_matplotlib_missing_raises_clear_import_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("missing matplotlib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(
        ImportError,
        match="Plotting requires matplotlib. Install with: pip install matplotlib",
    ):
        plot_experiment_results(result())


@pytest.mark.parametrize("metric_type", ["bad", "RELATIVE"])
def test_invalid_metric_type(metric_type):
    with pytest.raises(ValueError, match="metric_type"):
        plot_experiment_results(result(), metric_type=metric_type)


def test_empty_results_raise_value_error():
    with pytest.raises(ValueError, match="at least one treatment effect"):
        plot_experiment_results(FakeResults())


def test_custom_ax_is_used():
    fig, ax = plt.subplots()
    returned = plot_experiment_results(result(), ax=ax)
    assert returned is ax
