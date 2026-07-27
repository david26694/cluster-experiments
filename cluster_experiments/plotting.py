import math


def _import_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install with: pip install matplotlib"
        ) from e
    return plt


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _get(values, index, default=math.nan):
    return values[index] if index < len(values) else default


def _safe_float(value):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _relative(value, control_mean):
    value = _safe_float(value)
    control_mean = _safe_float(control_mean)
    if math.isnan(value) or math.isnan(control_mean):
        return math.nan
    if control_mean == 0:
        if value == 0:
            return math.nan
        return math.copysign(math.inf, value)
    return value / control_mean * 100


def _color(lift, p_value, alpha):
    lift = _safe_float(lift)
    p_value = _safe_float(p_value)
    alpha = _safe_float(alpha)
    if math.isnan(lift) or math.isnan(p_value) or math.isnan(alpha) or p_value >= alpha:
        return "#7F8C8D"
    if lift > 0:
        return "#2ECC71"
    if lift < 0:
        return "#E74C3C"
    return "#7F8C8D"


def _icon(lift):
    lift = _safe_float(lift)
    if math.isnan(lift):
        return "\u25cf"
    if lift > 0:
        return "\u25b2"
    if lift < 0:
        return "\u25bc"
    return "\u25cf"


def _format_lift(lift, suffix):
    lift = _safe_float(lift)
    if math.isnan(lift):
        return "nan"
    if math.isinf(lift):
        return "+inf" if lift > 0 else "-inf"
    return f"{lift:+.2f}{suffix}"


def _format_mean(value):
    value = _safe_float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.2f}"


def _ci_half_range(lows, highs, lifts):
    values = []
    for sequence in (lows, highs, lifts):
        for value in sequence:
            v = _safe_float(value)
            if math.isfinite(v):
                values.append(abs(v))
    if not values:
        return 1.0
    return max(max(values) * 1.5, 1.0)


def plot_experiment_results(result, metric_type="relative", title=None, ax=None):
    """Scoreboard-style experiment results plot.

    One horizontal "card" per treatment vs control: control circle (mean),
    treatment circle (mean), lift indicator (icon + value, colored by
    significance), and a centered CI bar with ticks. Multi-variant
    experiments stack rows in a single figure.

    Parameters
    ----------
    result
        Object exposing AnalysisPlanResults-like attributes: control_variant_mean,
        treatment_variant_mean, ate, ate_ci_lower, ate_ci_upper, p_value,
        and optionally alpha, control_variant_name, treatment_variant_name.
    metric_type : {"relative", "absolute"}
    title : str, optional
    ax : matplotlib.axes.Axes, optional
        Existing axes. If omitted, a new figure is created sized to the row count.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if metric_type not in {"relative", "absolute"}:
        raise ValueError("metric_type must be 'relative' or 'absolute'")

    plt = _import_pyplot()

    control_means = _as_list(getattr(result, "control_variant_mean", []))
    treatment_means = _as_list(getattr(result, "treatment_variant_mean", []))
    ates = _as_list(getattr(result, "ate", []))
    ci_lowers = _as_list(getattr(result, "ate_ci_lower", []))
    ci_uppers = _as_list(getattr(result, "ate_ci_upper", []))
    p_values = _as_list(getattr(result, "p_value", []))
    alphas = _as_list(getattr(result, "alpha", []))
    control_names = _as_list(getattr(result, "control_variant_name", []))
    treatment_names = _as_list(getattr(result, "treatment_variant_name", []))

    n_rows = len(ates)
    if n_rows == 0:
        raise ValueError("result must contain at least one treatment effect in ate")

    suffix = "%" if metric_type == "relative" else ""
    rows = []
    for index in range(n_rows):
        control_mean = _get(control_means, index)
        treatment_mean = _get(treatment_means, index)
        if metric_type == "relative":
            lift = _relative(_get(ates, index), control_mean)
            low = _relative(_get(ci_lowers, index), control_mean)
            high = _relative(_get(ci_uppers, index), control_mean)
            ctrl_display = (
                _safe_float(control_mean) * 100
                if not math.isnan(_safe_float(control_mean))
                else math.nan
            )
            treat_display = (
                _safe_float(treatment_mean) * 100
                if not math.isnan(_safe_float(treatment_mean))
                else math.nan
            )
        else:
            lift = _safe_float(_get(ates, index))
            low = _safe_float(_get(ci_lowers, index))
            high = _safe_float(_get(ci_uppers, index))
            ctrl_display = _safe_float(control_mean)
            treat_display = _safe_float(treatment_mean)

        rows.append(
            {
                "control_name": _get(control_names, index, "A"),
                "treatment_name": _get(treatment_names, index, chr(ord("B") + index)),
                "control_display": ctrl_display,
                "treatment_display": treat_display,
                "lift": lift,
                "low": low,
                "high": high,
                "color": _color(lift, _get(p_values, index), _get(alphas, index, 0.05)),
            }
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(0.8, 0.9 * n_rows)))

    ax.set_xlim(0, 11)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    half_range = _ci_half_range(
        [row["low"] for row in rows],
        [row["high"] for row in rows],
        [row["lift"] for row in rows],
    )
    bar_center = 8.0
    bar_width = 4.0

    for index, row in enumerate(rows):
        y_center = n_rows - index - 0.5
        y_low = y_center - 0.375
        y_high = y_center + 0.375
        y_top = y_center + 0.475

        ax.add_patch(
            plt.Circle(
                (0.3, y_center),
                0.2,
                color="#E8E8F0",
                ec="#B0B0C0",
                lw=2,
            )
        )
        ax.text(
            0.3,
            y_center,
            row["control_name"][:1] or "A",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#666",
        )
        ax.text(
            0.3,
            y_top,
            "control",
            ha="center",
            fontsize=8,
            color="#888",
        )
        ax.text(
            0.7,
            y_center,
            _format_mean(row["control_display"]),
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.add_patch(
            plt.Circle(
                (2.6, y_center),
                0.2,
                color="white",
                ec="#9B59B6",
                lw=2,
            )
        )
        ax.text(
            2.6,
            y_center,
            row["treatment_name"][:1] or "B",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#9B59B6",
        )
        ax.text(
            2.6,
            y_top,
            row["treatment_name"],
            ha="center",
            fontsize=8,
            color="#888",
        )
        ax.text(
            3.0,
            y_center,
            _format_mean(row["treatment_display"]),
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        ax.text(5.0, y_top, "Lift", ha="center", fontsize=8, color="#888")
        ax.text(
            5.0,
            y_center,
            f"{_icon(row['lift'])} {_format_lift(row['lift'], suffix)}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=row["color"],
        )

        ax.axvline(
            bar_center,
            ymin=(y_low) / n_rows,
            ymax=(y_high) / n_rows,
            color="#333",
            ls="--",
            lw=1,
            alpha=0.5,
        )
        ax.text(
            bar_center - bar_width / 2,
            y_top,
            f"{-half_range:.1f}{suffix}",
            ha="center",
            fontsize=7,
            color="#999",
        )
        ax.text(bar_center, y_top, f"0{suffix}", ha="center", fontsize=7, color="#999")
        ax.text(
            bar_center + bar_width / 2,
            y_top,
            f"+{half_range:.1f}{suffix}",
            ha="center",
            fontsize=7,
            color="#999",
        )

        low = _safe_float(row["low"])
        high = _safe_float(row["high"])
        if math.isfinite(low) and math.isfinite(high):
            clipped_low = max(min(low, half_range), -half_range)
            clipped_high = max(min(high, half_range), -half_range)
            x1 = bar_center + (clipped_low / half_range) * (bar_width / 2)
            x2 = bar_center + (clipped_high / half_range) * (bar_width / 2)
            ax.add_patch(
                plt.Rectangle(
                    (x1, y_center - 0.15),
                    x2 - x1,
                    0.3,
                    fc="#D0D0D8",
                    ec="none",
                    zorder=1,
                )
            )

        lift_value = _safe_float(row["lift"])
        if math.isfinite(lift_value):
            clipped = max(min(lift_value, half_range), -half_range)
            x_marker = bar_center + (clipped / half_range) * (bar_width / 2)
            ax.plot(x_marker, y_center, "o", color="#555", ms=6, zorder=2)
        else:
            ax.plot(bar_center, y_center, "x", color=row["color"], ms=6, zorder=2)

    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

    return ax
