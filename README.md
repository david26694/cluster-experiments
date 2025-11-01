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

### **Experiment Design**
- **Power Analysis & Sample Size Calculation**
  - Simulation-based (Monte Carlo) for any design complexity
  - Analytical, (CLT-based) for standard designs
  - Minimal Detectable Effect (MDE) estimation
  
- **Multiple Experimental Designs**
  - Standard A/B tests with individual randomization
  - Cluster-randomized experiments
  - Switchback/crossover experiments
  - Stratified randomization
  - Observational studies with Synthetic Control

### **Statistical Methods**
- **Multiple Analysis Methods**
  - OLS and Clustered OLS regression
  - GEE (Generalized Estimating Equations)
  - Mixed Linear Models (MLM)
  - Delta Method for ratio metrics
  - Synthetic Control for observational data

- **Variance Reduction Techniques**
  - CUPED (Controlled-experiment Using Pre-Experiment Data)
  - CUPAC (CUPED with Pre-experiment Aggregations)
  - Covariate adjustment

### **Analysis Workflow**
- **Scorecard Generation**: Analyze multiple metrics simultaneously
- **Multi-dimensional Slicing**: Break down results by segments
- **Multiple Treatment Arms**: Compare several treatments at once
- **Ratio Metrics**: Built-in support for conversion rates, averages, etc.

---

## 📦 Installation

```bash
pip install cluster-experiments
```

---

## ⚡ Quick Example

Here's a simple example showing how to analyze an experiment with multiple metrics organized by category - a common production pattern:

```python
import pandas as pd
import numpy as np
from cluster_experiments import (
    AnalysisPlan, SimpleMetric, RatioMetric, 
    Variant, HypothesisTest
)

# Simulate experiment data
np.random.seed(42)
n_users = 5000

data = pd.DataFrame({
    'user_id': range(n_users),
    'variant': np.random.choice(['control', 'treatment'], n_users),
    'orders': np.random.poisson(2.5, n_users),  # Number of orders
    'visits': np.random.poisson(10, n_users),   # Number of visits
})

# Add treatment effect: +20% orders for treatment
data.loc[data['variant'] == 'treatment', 'orders'] += \
    np.random.poisson(0.5, (data['variant'] == 'treatment').sum())

# Prepare data
data['converted'] = data['orders'].astype(int)

# Define metrics by type and category
absolute_metrics = {
    "orders": "revenue"  # metric_name: category
}

ratio_metrics = {
    "conversion_rate": {
        "category": "conversion",
        "components": ["converted", "visits"]  # [numerator, denominator]
    }
}

# Define variants
variants = [
    Variant("control", is_control=True),
    Variant("treatment", is_control=False)
]

# Build hypothesis tests from metric definitions
hypothesis_tests = []

# 1. Ratio metrics: use delta method for proper ratio analysis
for metric_name, config in ratio_metrics.items():
    metric = RatioMetric(
        alias=f"{config['category']}__{metric_name}",
        numerator_name=config['components'][0],
        denominator_name=config['components'][1]
    )
    hypothesis_tests.append(
        HypothesisTest(
            metric=metric,
            analysis_type="delta",
            analysis_config={
                "scale_col": metric.denominator_name,
                "cluster_cols": ["user_id"]
            }
        )
    )

# 2. Absolute metrics: use standard OLS
for metric_name, category in absolute_metrics.items():
    metric = SimpleMetric(
        alias=f"{category}__{metric_name}",
        name=metric_name
    )
    hypothesis_tests.append(
        HypothesisTest(
            metric=metric,
            analysis_type="ols"
        )
    )

# Create and run analysis plan
analysis_plan = AnalysisPlan(
    tests=hypothesis_tests,
    variants=variants,
    variant_col='variant'
)

results = analysis_plan.analyze(data, verbose=True)
results_df = results.to_dataframe()
print(results_df)
```

**Output**: A comprehensive scorecard with treatment effects, confidence intervals, and p-values:

```
  metric_alias                    control  treatment    ate   p_value  ...
  conversion__conversion_rate      0.250     0.303   +20.9%   < 0.001  ...
  revenue__orders                  2.510     3.005   +19.7%   < 0.001  ...
```

This example demonstrates:
- ✅ **Organized metric definitions** - Group metrics by type and category
- ✅ **Multiple analysis methods** - Delta method for ratios, OLS for totals
- ✅ **Scalable** - Easy to add more metrics by updating dictionaries

---

## 📚 Documentation

For detailed guides, API references, and advanced examples, visit our [**documentation**](https://david26694.github.io/cluster-experiments/).

### Key Resources
- [**Quickstart Guide**](https://david26694.github.io/cluster-experiments/quickstart.html): Get up and running in minutes
- [**API Reference**](https://david26694.github.io/cluster-experiments/api/experiment_analysis.html): Detailed class and method documentation
- [**Example Gallery**](https://david26694.github.io/cluster-experiments/cupac_example.html): Real-world use cases and patterns

---

##  Core Concepts

The library is built around three main components:

### 1. **Splitter** - Define how to randomize
Choose how to split your data into control and treatment groups:
- `NonClusteredSplitter`: Standard individual-level randomization
- `ClusteredSplitter`: Cluster-level randomization
- `SwitchbackSplitter`: Time-based alternating treatments
- `StratifiedClusteredSplitter`: Balance randomization across strata

### 2. **Analysis** - Measure the impact
Select the appropriate statistical method for your design:
- `OLSAnalysis`: Standard regression for A/B tests
- `ClusteredOLSAnalysis`: Clustered standard errors for cluster-randomized designs
- `TTestClusteredAnalysis`: T-tests on cluster-aggregated data
- `GeeExperimentAnalysis`: GEE for correlated observations
- `SyntheticControlAnalysis`: Observational studies with synthetic controls

### 3. **AnalysisPlan** - Orchestrate your analysis
Define your complete analysis workflow:
- Specify metrics (simple and ratio)
- Define variants and dimensions
- Configure hypothesis tests
- Generate comprehensive scorecards

For **power analysis**, combine these with:
- **Perturbator**: Simulate treatment effects for power calculations
- **PowerAnalysis**: Estimate statistical power and sample sizes

---

## When to Use cluster-experiments

✅ **Use cluster-experiments when you need to:**
- Design and analyze **cluster-randomized experiments**
- Handle **switchback/crossover designs**
- Account for **network effects or spillover**
- Perform **power analysis** for complex designs
- Reduce variance with **CUPED/CUPAC**
- Analyze **multiple metrics** with dimensional slicing
- Work with **ratio metrics** (rates, averages, etc.)

 **Perfect for:**
 - A/B tests
- Marketplace/platform experiments (drivers, restaurants, stores,...)
- Geographic experiments (cities, regions)
- Time-based tests (switchbacks, dayparting)

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

# 4. MDE timeline: How MDE changes with experiment length
mde_timeline = power_analysis.mde_time_line(
    historical_data,
    powers=[0.8],
    experiment_length=[7, 14, 21, 30]
)
```

**Output:**
```
Power for detecting +5 unit effect: 81.1%
Minimum detectable effect at 80% power: 4.93

Power Curve:
  effect  power
    2.0   20.6%
    4.0   62.2%
    6.0   92.6%
    8.0   99.5%
   10.0  100.0%

MDE Timeline (experiment length → MDE):
   7 days: 10.64
  14 days:  7.62
  21 days:  6.14
  30 days:  4.93
```

**Key methods:**
- `power_analysis()`: Calculate power for a given effect
- `mde()`: Calculate minimum detectable effect
- `power_line()`: Generate power curves across effect sizes
- `mde_time_line()`: Calculate MDE for different experiment lengths

For simulation-based power analysis (for complex designs), see the [Power Analysis Guide](https://david26694.github.io/cluster-experiments/power_analysis_guide.html).

---

## 🤝 Contributing

We welcome contributions! See our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:
- Report bugs
- Suggest features
- Submit pull requests
- Write documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
