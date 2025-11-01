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

`cluster-experiments` provides a complete toolkit for designing, running, and analyzing experiments, with particular strength in handling **clustered randomization** and complex experimental designs. Originally developed to address challenges in **switchback experiments** and scenarios with **network effects** where standard randomization isn't feasible, it has evolved into a general-purpose experimentation framework supporting both simple A/B tests and sophisticated designs.

### Why "cluster"?

The name reflects the library's origins in handling **cluster-randomized experiments**, where randomization happens at a group level (e.g., stores, cities, time periods) rather than at the individual level. This is critical when:

- **Spillover/Network Effects**: Treatment of one unit affects others (e.g., testing driver incentives in ride-sharing)
- **Operational Constraints**: You can't randomize individuals (e.g., testing restaurant menu changes)
- **Switchback Designs**: Treatment alternates over time periods within the same unit

While the library excels at these complex scenarios, it's equally capable of handling standard A/B tests with individual-level randomization.

---

## 🚀 Key Features

### 📊 **Comprehensive Experiment Design**
- **Power Analysis & Sample Size Calculation**
  - Simulation-based (Monte Carlo) for any design complexity
  - Analytical (CLT-based) for standard designs
  - Minimal Detectable Effect (MDE) estimation
  
- **Multiple Experimental Designs**
  - Standard A/B tests with individual randomization
  - Cluster-randomized experiments
  - Switchback/crossover experiments
  - Stratified randomization
  - Observational studies with Synthetic Control

### 🔬 **Advanced Statistical Methods**
- **Multiple Analysis Methods**
  - OLS and Clustered OLS regression
  - T-tests and Paired T-tests
  - GEE (Generalized Estimating Equations)
  - Mixed Linear Models (MLM)
  - Delta Method for ratio metrics
  - Synthetic Control for observational data

- **Variance Reduction Techniques**
  - CUPED (Controlled-experiment Using Pre-Experiment Data)
  - CUPAC (CUPED with Pre-experiment Aggregations)
  - Covariate adjustment

### 📈 **Scalable Analysis Workflow**
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

Here's a simple example showing how to analyze an experiment with two metrics: a simple metric (conversions) and a ratio metric (conversion rate).

```python
import pandas as pd
import numpy as np
from cluster_experiments import AnalysisPlan

# Simulate experiment data
np.random.seed(42)
n_users = 1000

data = pd.DataFrame({
    'user_id': range(n_users),
    'variant': np.random.choice(['control', 'treatment'], n_users),
    'orders': np.random.poisson(2.5, n_users),  # Number of orders (simple metric)
    'visits': np.random.poisson(10, n_users),   # Number of visits (for ratio)
})

# Add a small treatment effect to orders
data.loc[data['variant'] == 'treatment', 'orders'] += np.random.poisson(0.5, (data['variant'] == 'treatment').sum())

# Calculate conversions (users who ordered)
data['converted'] = (data['orders'] > 0).astype(int)

# Define analysis plan
analysis_plan = AnalysisPlan.from_metrics_dict({
    'metrics': [
        # Simple metric: total conversions
        {
            'alias': 'conversions',
            'name': 'converted',
            'metric_type': 'simple'
        },
        # Ratio metric: conversion rate (conversions / visits)
        {
            'alias': 'conversion_rate', 
            'metric_type': 'ratio',
            'numerator': 'converted',
            'denominator': 'visits'
        },
    ],
    'variants': [
        {'name': 'control', 'is_control': True},
        {'name': 'treatment', 'is_control': False},
    ],
    'variant_col': 'variant',
    'analysis_type': 'ols',  # Use OLS for simple A/B test
})

# Run analysis
results = analysis_plan.analyze(data)

# View results as a dataframe
print(results.to_dataframe())
```

**Output**: A comprehensive scorecard with treatment effects, confidence intervals, and p-values for each metric:

```
        metric  control_mean  treatment_mean  ...  p_value  ci_lower  ci_upper
0  conversions         0.485           0.532  ...    0.023     0.006     0.088
1  conversion_rate     0.048           0.053  ...    0.031     0.0004    0.009
```

This simple example demonstrates:
- ✅ Working with both **simple** and **ratio metrics**
- ✅ Easy experiment setup with **dictionary-based configuration**
- ✅ Statistical inference with **confidence intervals and p-values**
- ✅ **Automatic scorecard generation** for multiple metrics

---

## 📚 Documentation

For detailed guides, API references, and advanced examples, visit our [**documentation**](https://david26694.github.io/cluster-experiments/).

### Key Resources
- [**Quickstart Guide**](https://david26694.github.io/cluster-experiments/quickstart.html): Get up and running in minutes
- [**API Reference**](https://david26694.github.io/cluster-experiments/api/experiment_analysis.html): Detailed class and method documentation
- [**Example Gallery**](https://david26694.github.io/cluster-experiments/cupac_example.html): Real-world use cases and patterns

---

## 🎯 Core Concepts

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

## 🔍 When to Use cluster-experiments

✅ **Use cluster-experiments when you need to:**
- Design and analyze **cluster-randomized experiments**
- Handle **switchback/crossover designs**
- Account for **network effects or spillover**
- Perform **power analysis** for complex designs
- Reduce variance with **CUPED/CUPAC**
- Analyze **multiple metrics** with dimensional slicing
- Work with **ratio metrics** (rates, averages, etc.)

📊 **Perfect for:**
- Marketplace/platform experiments (drivers, restaurants, stores)
- Geographic experiments (cities, regions)
- Time-based tests (switchbacks, dayparting)
- Standard A/B tests with advanced analysis needs

---

## 🛠️ Advanced Features

### Variance Reduction with CUPAC

Reduce variance by leveraging pre-experiment data:

```python
from cluster_experiments import AnalysisPlan, TargetAggregation, HypothesisTest, SimpleMetric, Variant

# Define CUPAC model
cupac_model = TargetAggregation(
    agg_col="customer_id",
    target_col="order_value"
)

# Create hypothesis test with CUPAC
test = HypothesisTest(
    metric=SimpleMetric(alias="revenue", name="order_value"),
    analysis_type="clustered_ols",
    analysis_config={
        "cluster_cols": ["customer_id"],
        "covariates": ["customer_age", "estimate_order_value"],
    },
    cupac_config={
        "cupac_model": cupac_model,
        "target_col": "order_value",
    },
)

plan = AnalysisPlan(
    tests=[test],
    variants=[Variant("control", is_control=True), Variant("treatment")],
    variant_col="variant",
)

# Analyze with pre-experiment data
results = plan.analyze(experiment_df, pre_experiment_df)
```

### Power Analysis

Estimate the power of your experiment design:

```python
from cluster_experiments import PowerAnalysis, NormalPowerAnalysis
from cluster_experiments import ClusteredSplitter, ConstantPerturbator, ClusteredOLSAnalysis

# Simulation-based power analysis
power_sim = PowerAnalysis(
    splitter=ClusteredSplitter(cluster_cols=['city']),
    perturbator=ConstantPerturbator(average_effect=0.1),
    analysis=ClusteredOLSAnalysis(cluster_cols=['city']),
    n_simulations=1000
)

power = power_sim.power_analysis(historical_data, average_effect=0.1)
print(f"Estimated power: {power:.2%}")

# Analytical power analysis (faster for standard designs)
power_analytical = NormalPowerAnalysis.from_dict({
    'cluster_cols': ['city'],
    'analysis': 'clustered_ols'
})

mde = power_analytical.mde(historical_data, power=0.8)
print(f"Minimum Detectable Effect at 80% power: {mde:.4f}")
```

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
