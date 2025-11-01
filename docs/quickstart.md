# Quickstart

Get started with `cluster-experiments` in minutes! This guide will walk you through installation and your first experiment analysis.

---

## Installation

Install via pip:

```bash
pip install cluster-experiments
```

!!! info "Requirements"
    - **Python 3.8 or higher**
    - Main dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`

---

## Your First Analysis (5 minutes)

Let's analyze a simple A/B test with multiple metrics. This is the most common use case.

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
    'orders': np.random.poisson(2.5, n_users),
    'visits': np.random.poisson(10, n_users),
})

# Add treatment effect
data.loc[data['variant'] == 'treatment', 'orders'] += np.random.poisson(0.5, (data['variant'] == 'treatment').sum())
data['converted'] = (data['orders'] > 0).astype(int)

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

# Run analysis
results = analysis_plan.analyze(data)
print(results.to_dataframe())
```

**Output:** A comprehensive scorecard with treatment effects, confidence intervals, and p-values!

---

## Understanding Your Results

The results dataframe includes:

| Column | Description |
|--------|-------------|
| `metric` | Name of the metric being analyzed |
| `control_mean` | Average value in control group |
| `treatment_mean` | Average value in treatment group |
| `ate` | Average Treatment Effect (absolute difference) |
| `ate_ci_lower/upper` | 95% confidence interval for ATE |
| `p_value` | Statistical significance (< 0.05 = significant) |

!!! tip "Interpreting Results"
    - **p_value < 0.05**: Result is statistically significant
    - **Confidence interval**: If it doesn't include 0, effect is significant

---

## Common Use Cases

### 1. Analyzing an Experiment

**When:** You've already run your experiment and have the data.

**Example:** See [Simple A/B Test](examples/simple_ab_test.html) for a complete walkthrough.

```python
# Use AnalysisPlan with your experiment data
results = analysis_plan.analyze(experiment_data)
```

---

### 2. Power Analysis (Sample Size Planning)

**When:** You're designing an experiment and need to know how many users/time you need.

**Example:** Calculate power or Minimum Detectable Effect (MDE).

```python
import numpy as np
import pandas as pd
from cluster_experiments import NormalPowerAnalysis

# Create historical data
np.random.seed(42)
historical_data = pd.DataFrame({
    'user_id': range(500),
    'metric': np.random.normal(100, 20, 500),
    'date': pd.to_datetime('2025-10-01') + pd.to_timedelta(np.random.randint(0, 30, 500), unit='d')
})

# Define your analysis setup
power_analysis = NormalPowerAnalysis.from_dict({
    'analysis': 'ols',
    'splitter': 'non_clustered',
    'target_col': 'metric',
    'time_col': 'date'
})

# Calculate MDE for 80% power
mde = power_analysis.mde(historical_data, power=0.8)
print(f"Need {mde:.2f} effect size for 80% power")

# Or calculate power for a given effect size
power = power_analysis.power_analysis(historical_data, average_effect=5.0)
print(f"Power: {power:.1%}")
```

**Learn more:** See [Power Analysis Guide](power_analysis_guide.html) for detailed explanation.

---

### 3. Cluster Randomization

**When:** Randomization happens at group level (stores, cities) rather than individual level.

**Why:** Required when there are spillover effects or operational constraints.

**Example:**

```python
import pandas as pd
import numpy as np
from cluster_experiments import AnalysisPlan

# Simulate store-level experiment data
np.random.seed(42)
n_stores = 50
transactions_per_store = 100

data = []
for store_id in range(n_stores):
    variant = np.random.choice(['control', 'treatment'])
    n_trans = np.random.poisson(transactions_per_store)
    
    store_data = pd.DataFrame({
        'store_id': store_id,
        'variant': variant,
        'purchase_amount': np.random.normal(50, 20, n_trans)
    })
    data.append(store_data)

experiment_data = pd.concat(data, ignore_index=True)

# Use clustered_ols for cluster-randomized experiments
analysis_plan = AnalysisPlan.from_metrics_dict({
    'metrics': [{'alias': 'revenue', 'name': 'purchase_amount', 'metric_type': 'simple'}],
    'variants': [
        {'name': 'control', 'is_control': True},
        {'name': 'treatment', 'is_control': False},
    ],
    'variant_col': 'variant',
    'analysis_type': 'clustered_ols',  # ← Key difference!
    'analysis_config': {
        'cluster_cols': ['store_id']  # ← Specify clustering variable
    }
})

results = analysis_plan.analyze(experiment_data)
print(results.to_dataframe())
```

**Learn more:** See [Cluster Randomization Example](examples/cluster_randomization.html).

---

### 4. Variance Reduction (CUPAC/CUPED)

**When:** You have pre-experiment data and want to reduce variance for more sensitive tests.

**Benefits:** Detect smaller effects with same sample size.

**Example:**

```python
import pandas as pd
import numpy as np
from cluster_experiments import (
    AnalysisPlan, TargetAggregation, HypothesisTest, 
    SimpleMetric, Variant
)

# Simulate experiment data
np.random.seed(42)
n_customers = 1000

experiment_data = pd.DataFrame({
    'customer_id': range(n_customers),
    'variant': np.random.choice(['control', 'treatment'], n_customers),
    'order_value': np.random.normal(100, 20, n_customers),
    'customer_age': np.random.randint(20, 60, n_customers),
})

# Simulate pre-experiment data (historical)
pre_experiment_data = pd.DataFrame({
    'customer_id': range(n_customers),
    'order_value': np.random.normal(95, 25, n_customers),  # Historical order values
})

# Define CUPAC model using pre-experiment data
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
    variants=[Variant("control", is_control=True), Variant("treatment", is_control=False)],
    variant_col="variant",
)

# Analyze with both experiment and pre-experiment data
results = plan.analyze(experiment_data, pre_experiment_data)
print(results.to_dataframe())
```

**Learn more:** See [CUPAC Example](cupac_example.html).

---

## Ratio Metrics

`cluster-experiments` has built-in support for ratio metrics (e.g., conversion rate, average order value), as seen in the first example:

```python
# Ratio metric: conversions / visits
{
    'alias': 'conversion_rate',
    'metric_type': 'ratio',
    'numerator_name': 'converted',      # Numerator column
    'denominator_name': 'visits'         # Denominator column
}
```

The library automatically handles the statistical complexities of ratio metrics using the Delta Method.

---

## Multi-Dimensional Analysis

Slice your results by dimensions (e.g., city, device type):

```python
analysis_plan = AnalysisPlan.from_metrics_dict({
    'metrics': [...],
    'variants': [...],
    'variant_col': 'variant',
    'dimensions': [
        {'name': 'city', 'values': ['NYC', 'LA', 'Chicago']},
        {'name': 'device', 'values': ['mobile', 'desktop']},
    ],
    'analysis_type': 'ols',
})
```

Results will include treatment effects for each dimension slice!

---

## Quick Reference

### Analysis Types

Choose the appropriate analysis method:

| Analysis Type | When to Use |
|--------------|-------------|
| `ols` | Standard A/B test, individual randomization |
| `clustered_ols` | Cluster randomization (stores, cities, etc.) |
| `gee` | Repeated measures, correlated observations |
| `mlm` | Multi-level/hierarchical data |
| `synthetic_control` | Observational studies, no randomization |

### Dictionary vs Class-Based API

Two ways to define analysis plans:

**Dictionary (simpler):**
```python
plan = AnalysisPlan.from_metrics_dict({...})
```

**Class-based (more control):**
```python
from cluster_experiments import HypothesisTest, SimpleMetric, Variant

plan = AnalysisPlan(
    tests=[HypothesisTest(metric=SimpleMetric(...), ...)],
    variants=[Variant(...)],
    variant_col='variant'
)
```

---

## Next Steps

Now that you've completed your first analysis, explore:

- 📖 **[API Reference](api/experiment_analysis.html)** - Detailed documentation for all classes
- **[Example Gallery](cupac_example.html)** - Real-world use cases and patterns
- **[Power Analysis Guide](power_analysis_guide.html)** - Design experiments with confidence
- 🤝 **[Contributing](../CONTRIBUTING.md)** - Help improve the library

---

## Getting Help

- 📝 [Documentation](https://david26694.github.io/cluster-experiments/)
- 🐛 [Report Issues](https://github.com/david26694/cluster-experiments/issues)
- 💬 [Discussions](https://github.com/david26694/cluster-experiments/discussions)
