<img src="theme/icon-cluster.png" width=200 height=200 align="right">

# cluster-experiments

[![Downloads](https://static.pepy.tech/badge/cluster-experiments)](https://pepy.tech/project/cluster-experiments)
[![PyPI](https://img.shields.io/pypi/v/cluster-experiments)](https://pypi.org/project/cluster-experiments/)
[![Unit tests](https://github.com/david26694/cluster-experiments/workflows/Release%20unit%20Tests/badge.svg)](https://github.com/david26694/cluster-experiments/actions)
[![CodeCov](https://codecov.io/gh/david26694/cluster-experiments/branch/main/graph/badge.svg)](https://app.codecov.io/gh/david26694/cluster-experiments/)
![License](https://img.shields.io/github/license/david26694/cluster-experiments)
[![Pypi version](https://img.shields.io/pypi/pyversions/cluster-experiments.svg)](https://pypi.python.org/pypi/cluster-experiments)

**`cluster-experiments`** is a comprehensive Python library for **end-to-end A/B testing workflows**, from experiment design to statistical analysis.

## 📖 What is cluster-experiments?

`cluster-experiments` provides a complete toolkit for designing, running, and analyzing experiments, with particular strength in handling **clustered randomization** and complex experimental designs. Originally developed to address challenges in **switchback experiments** and scenarios with **network effects** where standard randomization isn't feasible, it has evolved into a general-purpose experimentation framework supporting both simple A/B tests and other randomization designs.

### Why "cluster"?

The name reflects the library's origins in handling **cluster-randomized experiments**, where randomization happens at a group level (e.g., stores, cities, time periods) rather than at the individual level. This is critical when:

- **Spillover/Network Effects**: Treatment of one unit affects others (e.g., testing driver incentives in ride-sharing)
- **Operational Constraints**: You can't randomize individuals (e.g., testing restaurant menu changes)
- **Switchback Designs**: Treatment alternates over time periods within the same unit

While the library is aimed at these scenarios, it's equally capable of handling standard A/B tests with individual-level randomization.

---

## Key Features

### Experiment Design

<details markdown="1">
<summary>Power Analysis & Sample Size Calculation</summary>

- Simulation-based (Monte Carlo) for any design complexity
- Analytical (CLT-based) for standard designs
- Minimum Detectable Effect (MDE) estimation
</details>

<details markdown="1">
<summary>Multiple Experimental Designs</summary>

- Standard A/B tests with individual randomization
- Cluster-randomized experiments
- Switchback/crossover experiments
- Stratified randomization
- Observational studies with Synthetic Control
</details>

### Statistical Methods

<details markdown="1">
<summary>Multiple Analysis Methods</summary>

- OLS and Clustered OLS regression
- GEE (Generalized Estimating Equations)
- Mixed Linear Models (MLM)
- Delta Method for ratio metrics
- Synthetic Control for observational data
</details>

<details markdown="1">
<summary>Variance Reduction Techniques</summary>

- CUPED (Controlled-experiment Using Pre-Experiment Data)
- CUPAC (Control Using Predictions As Covariates)
- Covariate adjustment
</details>

### Analysis Workflow

<details markdown="1">
<summary>Scorecard & Multi-dimensional Analysis</summary>

- **Scorecard Generation**: Analyze multiple metrics simultaneously
- **Multi-dimensional Slicing**: Break down results by segments
- **Multiple Treatment Arms**: Compare several treatments at once
- **Ratio Metrics**: Built-in support for conversion rates, averages, etc.
- **Relative Lift**: Analyze effects as percentage changes rather than absolute differences
</details>

---

## 📦 Installation

```bash
pip install cluster-experiments
```

---

## ⚡ Quick Example

Here's how to run an analysis in just a few lines:

```python
import pandas as pd
import numpy as np
from cluster_experiments import AnalysisPlan, Variant

np.random.seed(42)

# 0. Create simple data
N = 1_000
df = pd.DataFrame({
    "variant": np.random.choice(["control", "treatment"], N),
    "orders": np.random.poisson(10, N),
    "visits": np.random.poisson(100, N),
})
df["converted"] = (df["orders"] > 0).astype(int)


# 1. Define your analysis plan
plan = AnalysisPlan.from_metrics_dict({
    "metrics": [
        {"name": "orders", "alias": "revenue", "metric_type": "simple"},
        {"name": "converted", "alias": "conversion", "metric_type": "ratio", "numerator": "converted", "denominator": "visits"}
    ],
    "variants": [
        {"name": "control", "is_control": True},
        {"name": "treatment", "is_control": False}
    ],
    "variant_col": "variant",
    "analysis_type": "ols"
})

# 2. Run analysis on your dataframe
results = plan.analyze(df)
print(results.to_dataframe().head())
```

**Output Example**:

```
  metric_alias control_variant_name treatment_variant_name  control_variant_mean  treatment_variant_mean analysis_type           ate  ate_ci_lower  ate_ci_upper   p_value     std_error     dimension_name dimension_value  alpha
0      revenue              control              treatment              10.08554                9.941061           ols -1.444788e-01 -5.446603e-01  2.557026e-01  0.479186  2.041780e-01  __total_dimension           total   0.05
1   conversion              control              treatment               1.00000                1.000000           ols  1.110223e-16 -1.096504e-16  3.316950e-16  0.324097  1.125902e-16  __total_dimension           total   0.05
```

---

## Power Analysis

Design your experiment by estimating required sample size and detectable effects. Here's a complete example using **analytical (CLT-based) power analysis**:

```python
import numpy as np
import pandas as pd
from cluster_experiments import NormalPowerAnalysis

# Create sample historical data
np.random.seed(42)
N = 500

historical_data = pd.DataFrame({
    'user_id': range(N),
    'metric': np.random.normal(100, 20, N),
    'date': pd.to_datetime('2025-10-01') + pd.to_timedelta(np.random.randint(0, 30, N), unit='d')
})

# Initialize analytical power analysis (fast, CLT-based)
power_analysis = NormalPowerAnalysis.from_dict({
    'analysis': 'ols',
    'splitter': 'non_clustered',
    'target_col': 'metric',
    'time_col': 'date'  # Required for mde_time_line
})

# 1. Calculate power for a given effect size
power = power_analysis.power_analysis(historical_data, average_effect=5.0)
print(f"Power for detecting +5 unit effect: {power:.1%}")

# 2. Calculate Minimum Detectable Effect (MDE) for desired power
mde = power_analysis.mde(historical_data, power=0.8)
print(f"Minimum detectable effect at 80% power: {mde:.2f}")

# 3. Power curve: How power changes with effect size
power_curve = power_analysis.power_line(
    historical_data,
    average_effects=[2.0, 4.0, 6.0, 8.0, 10.0]
)
print(power_curve)

# 4. MDE timeline: How MDE changes with experiment length
mde_timeline = power_analysis.mde_time_line(
    historical_data,
    powers=[0.8],
    experiment_length=[7, 14, 21, 30]
)
```

**Output:**

```
Power for detecting +5 unit effect: 72.7%
Minimum detectable effect at 80% power: 5.46
{2.0: 0.18, 4.0: 0.54, 6.0: 0.87, 8.0: 0.98, 10.0: 1.00}
```

**Key methods:**

- `power_analysis()`: Calculate power for a given effect
- `mde()`: Calculate minimum detectable effect
- `power_line()`: Generate power curves across effect sizes
- `mde_time_line()`: Calculate MDE for different experiment lengths

For simulation-based power analysis (for complex designs), see the [Power Analysis Guide](https://david26694.github.io/cluster-experiments/normal_power_lines.html).

---

## 📚 Documentation

For detailed guides, API references, and advanced examples, visit our [**documentation**](https://david26694.github.io/cluster-experiments/).

### Core Concepts

The library is built around three main components:

#### 1. **Splitter** - Define how to randomize

Choose how to split your data into control and treatment groups:

- `NonClusteredSplitter`: Standard individual-level randomization
- `ClusteredSplitter`: Cluster-level randomization
- `SwitchbackSplitter`: Time-based alternating treatments
- `StratifiedClusteredSplitter`: Balance randomization across strata

#### 2. **Analysis** - Measure the impact

Select the appropriate statistical method for your design:

- `OLSAnalysis`: Standard regression for A/B tests
- `ClusteredOLSAnalysis`: Clustered standard errors for cluster-randomized designs
- `TTestClusteredAnalysis`: T-tests on cluster-aggregated data
- `GeeExperimentAnalysis`: GEE for correlated observations
- `SyntheticControlAnalysis`: Observational studies with synthetic controls

#### 3. **AnalysisPlan** - Orchestrate your analysis

Define your complete analysis workflow:

- Specify metrics (simple and ratio)
- Define variants and dimensions
- Configure hypothesis tests
- Generate comprehensive scorecards

For **power analysis**, combine these with:

- **Perturbator**: Simulate treatment effects for power calculations
- **PowerAnalysis**: Estimate statistical power and sample sizes

---

## 🛠️ Advanced Features

### Variance Reduction (CUPED/CUPAC)

Reduce variance and detect smaller effects by leveraging pre-experiment data. Use historical metrics as covariates to control for pre-existing differences between groups.

**Use cases:**

- Have pre-experiment metrics for your users/clusters
- Want to detect smaller treatment effects
- Need more sensitive tests with same sample size

See the [CUPAC Example](https://david26694.github.io/cluster-experiments/cupac_example.html) for detailed implementation.

### Cluster Randomization

Handle experiments where randomization occurs at group level (stores, cities, regions) rather than individual level. Essential for managing spillover effects and operational constraints.

See the [Cluster Randomization Guide](https://david26694.github.io/cluster-experiments/examples/cluster_randomization.html) for details.

### Switchback Experiments

Design and analyze time-based crossover experiments where the same units receive both control and treatment at different times.

See the [Switchback Example](https://david26694.github.io/cluster-experiments/switchback.html) for implementation.

---

## 🌟 Support

- ⭐ Star us on [GitHub](https://github.com/david26694/cluster-experiments)
- 📝 Read the [documentation](https://david26694.github.io/cluster-experiments/)
- 🐛 Report issues on our [issue tracker](https://github.com/david26694/cluster-experiments/issues)
- 💬 Join discussions in [GitHub Discussions](https://github.com/david26694/cluster-experiments/discussions)

---

## 📚 Citation

If you use cluster-experiments in your research, please cite:

```bibtex
@software{cluster_experiments,
  author = {David Masip and contributors},
  title = {cluster-experiments: A Python library for designing and analyzing experiments},
  url = {https://github.com/david26694/cluster-experiments},
  year = {2022}
}
```
