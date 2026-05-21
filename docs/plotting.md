# Plotting Experiment Results

`cluster-experiments` ships a single, dependency-light plotting helper -
`plot_experiment_results` - that turns an `AnalysisPlanResults` object into a
compact scoreboard-style chart suitable for experiment readouts.

It is intentionally tiny: one function, lazy `matplotlib` import, no themes,
no wrappers around seaborn or plotly.

---

## Installation

The plotter requires `matplotlib`, which is **not** a hard dependency of
`cluster-experiments`. Install it on demand:

```bash
pip install matplotlib
```

!!! info "Why optional?"
    Keeping `matplotlib` optional means the core library stays lightweight for
    pipelines and notebooks that only need analysis, not plots. The plotter
    raises a clear `ImportError` if `matplotlib` is missing.

---

## 1. Your First Plot

Run an analysis and plot the result in two lines:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cluster_experiments import AnalysisPlan, plot_experiment_results

np.random.seed(42)

# 1. Simulated experiment data
N = 5_000
df = pd.DataFrame({
    "variant": np.random.choice(["control", "treatment"], N),
    "orders":  np.random.poisson(10, N),
    "visits":  np.random.poisson(100, N),
})
df.loc[df["variant"] == "treatment", "orders"] += 1   # plant a real lift

# 2. Define the analysis
plan = AnalysisPlan.from_metrics_dict({
    "metrics": [
        {"name": "orders", "alias": "revenue", "metric_type": "simple"},
    ],
    "variants": [
        {"name": "control",   "is_control": True},
        {"name": "treatment", "is_control": False},
    ],
    "variant_col": "variant",
    "analysis_type": "ols",
})

# 3. Run analysis
results = plan.analyze(df)

# 4. Plot
plot_experiment_results(results, metric_type="relative", title="Revenue lift")
plt.show()
```

You get a single horizontal card with:

- a gray **control** circle and its mean
- a purple **treatment** circle and its mean
- a colored lift indicator (▲ green, ▼ red, ● gray)
- a centered CI bar with `-r% / 0% / +r%` ticks

---

## 2. Understanding the Mental Model

`plan.analyze(df)` returns an `AnalysisPlanResults` object whose attributes
are parallel lists - one entry per hypothesis test
(metric × variant × dimension slice).

| Plot row corresponds to | Example |
|---|---|
| One metric × one dimension × one treatment | 1 card |
| One metric × A/B/C/D | 3 cards (B vs A, C vs A, D vs A) |
| Three metrics × A/B | 3 cards (one per metric) |
| One metric × 3 cities × A/B | 3 cards (one per city) |

!!! tip "Rule of thumb"
    Filter the results down to **what belongs on a single chart**, then plot.
    A reusable helper is shown in section 5.

---

## 3. Relative vs Absolute

`plot_experiment_results` supports two `metric_type` modes:

| `metric_type` | Formula | When to use |
|---|---|---|
| `"relative"` | `ate / control_mean * 100` | Conversion rate, CTR, anything proportion-like |
| `"absolute"` | raw `ate` (metric units) | Revenue, durations, counts |

Side by side:

```python
fig, axes = plt.subplots(2, 1, figsize=(10, 5))
plot_experiment_results(results, metric_type="relative", ax=axes[0], title="Relative lift")
plot_experiment_results(results, metric_type="absolute", ax=axes[1], title="Absolute lift")
plt.tight_layout()
plt.show()
```

---

## 4. Multi-Variant Experiments (A/B/C/D)

Nothing changes - `AnalysisPlanResults` already contains one row per
non-control variant, and the plotter stacks one card per row in the same
figure.

```python
df["variant"] = np.random.choice(["control", "B", "C", "D"], N)

plan = AnalysisPlan.from_metrics_dict({
    "metrics": [{"name": "orders", "alias": "revenue", "metric_type": "simple"}],
    "variants": [
        {"name": "control", "is_control": True},
        {"name": "B",       "is_control": False},
        {"name": "C",       "is_control": False},
        {"name": "D",       "is_control": False},
    ],
    "variant_col": "variant",
    "analysis_type": "ols",
})

results = plan.analyze(df)
plot_experiment_results(results, metric_type="relative", title="A/B/C/D - revenue")
plt.show()
```

This produces three cards (B vs A, C vs A, D vs A) in one figure - no separate
figures per variant.

---

## 5. Filtering Results Before Plotting

When the analysis returns many rows (multiple metrics or dimensions), pick the
slice you want to chart:

```python
from cluster_experiments.inference.analysis_results import AnalysisPlanResults


def filter_results(results, metric_alias=None, dimension_value=None):
    """Return a new AnalysisPlanResults keeping only matching rows."""
    df = results.to_dataframe()
    if metric_alias is not None:
        df = df[df["metric_alias"] == metric_alias]
    if dimension_value is not None:
        df = df[df["dimension_value"] == dimension_value]
    return AnalysisPlanResults(**{c: df[c].tolist() for c in df.columns})


revenue = filter_results(results, metric_alias="revenue")
plot_experiment_results(revenue, metric_type="relative", title="Revenue lift")
plt.show()
```

---

## 6. Plotting With Dimensions

Dimensions multiply rows (one per `dimension_value`). Two common layouts:

### 6.1. One Chart per Dimension Slice

```python
for city in ["NYC", "LA", "Chicago"]:
    sliced = filter_results(results, metric_alias="revenue", dimension_value=city)
    plot_experiment_results(sliced, metric_type="relative", title=f"Revenue lift - {city}")
    plt.show()
```

### 6.2. All Slices in One Chart

The plotter labels each card with `treatment_variant_name`, so override it to
show the dimension value instead:

```python
by_city = filter_results(results, metric_alias="revenue")
by_city.treatment_variant_name = ["NYC", "LA", "Chicago"]
plot_experiment_results(by_city, metric_type="relative", title="Revenue lift by city")
plt.show()
```

---

## 7. Building a Scorecard (One Card per Metric)

A typical experiment readout shows every metric on its own row. Combine
filtering with a custom `ax`:

```python
metrics = ["revenue", "conversion"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 1.5 * len(metrics)))
for ax, metric in zip(axes, metrics):
    sliced = filter_results(results, metric_alias=metric)
    plot_experiment_results(sliced, metric_type="relative", ax=ax, title=metric)
plt.tight_layout()
plt.show()
```

---

## 8. Saving Plots (Headless / CI)

```python
import matplotlib
matplotlib.use("Agg")            # before importing pyplot
import matplotlib.pyplot as plt

ax = plot_experiment_results(results, metric_type="relative", title="Revenue")
ax.figure.savefig("revenue.png", dpi=120, bbox_inches="tight")
```

---

## 9. Edge Cases

The plotter handles common analysis quirks without crashing:

| Situation | Behavior |
|---|---|
| `control_variant_mean = 0` in relative mode | Card shows `+inf` / `-inf` label, gray **x** marker |
| `ate` or CI is `NaN` | Gray **x** marker, label shows `nan` |
| `p_value` missing or `NaN` | Treated as non-significant (gray) |
| `p_value >= alpha` | Gray (default `alpha = 0.05`) |
| Empty results | `ValueError("result must contain at least one treatment effect in ate")` |
| `matplotlib` not installed | `ImportError: Plotting requires matplotlib. Install with: pip install matplotlib` |

!!! tip "Custom significance level"
    Pass per-row `alpha` values on the result (e.g. `results.alpha = [0.01] * n`)
    to use a stricter threshold for coloring.

---

## 10. API Reference

::: cluster_experiments.plotting.plot_experiment_results

---

## Next Steps

- Explore the [Simple A/B Test](examples/simple_ab_test.html) walkthrough
- Read about [Relative Lift Analysis](relative.html) to understand the math behind `metric_type="relative"`
- See the [Analysis Plan API](api/analysis_plan.html) for full configuration options
