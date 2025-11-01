# Power Analysis Guide

This guide explains how to design experiments using **power analysis** to determine sample sizes and experiment duration.

---

## What is Power Analysis?

**Power analysis** helps you answer questions like:
- How many users do I need to detect a 5% lift?
- How long should I run my experiment?
- What's the smallest effect I can reliably detect?

This is done **before** running your experiment, using historical data to simulate different scenarios.

---

## When to Use Power Analysis

Use power analysis when:
- ✅ **Planning an experiment**: Determine required sample size
- ✅ **Evaluating feasibility**: Check if an effect is detectable with available data
- ✅ **Optimizing duration**: Balance statistical power with business needs

Don't need power analysis when:
- ❌ Analyzing completed experiments (use `AnalysisPlan` instead)
- ❌ You have unlimited sample size (though you should still check!)

---

## Two Approaches

### 1. Normal Approximation (Recommended)

**Pros:** Fast, uses analytical formulas  
**Cons:** Assumes normal distribution (works well for large samples)

```python
from cluster_experiments import NormalPowerAnalysis

power_analysis = NormalPowerAnalysis.from_dict({
    'analysis': 'ols',
    'splitter': 'non_clustered',
})

# Calculate MDE for 80% power
mde = power_analysis.mde(historical_data, power=0.8)
print(f"Minimum Detectable Effect: {mde:.2%}")

# Calculate power for specific effect size
power = power_analysis.power_analysis(historical_data, average_effect=0.05)
print(f"Power for 5% effect: {power:.1%}")
```

### 2. Simulation-Based

**Pros:** Works for any distribution, more accurate for complex designs  
**Cons:** Slower (runs many simulations)

```python
from cluster_experiments import PowerAnalysis
from cluster_experiments import ClusteredSplitter, ConstantPerturbator, ClusteredOLSAnalysis

# Define components
splitter = ClusteredSplitter(cluster_cols=['store_id'])
perturbator = ConstantPerturbator()  # Simulates treatment effect
analysis = ClusteredOLSAnalysis(cluster_cols=['store_id'])

# Create power analysis
power_analysis = PowerAnalysis(
    splitter=splitter,
    perturbator=perturbator,
    analysis=analysis,
    n_simulations=1000  # Number of simulations
)

# Run power analysis
power = power_analysis.power_analysis(historical_data, average_effect=0.1)
```

---

## Understanding Components (Simulation-Based)

For simulation-based power analysis, you need three components:

### 1. Splitter: How to Randomize

The **Splitter** defines how to divide your data into control and treatment groups.

#### Available Splitters

| Splitter | Use Case | Example |
|----------|----------|---------|
| `NonClusteredSplitter` | Individual-level randomization | User-level A/B test |
| `ClusteredSplitter` | Cluster-level randomization | Store-level test |
| `SwitchbackSplitter` | Time-based alternating treatment | Daily switchback |
| `StratifiedClusteredSplitter` | Balanced cluster randomization | Stratified by region |

#### Example

```python
from cluster_experiments import ClusteredSplitter

splitter = ClusteredSplitter(
    cluster_cols=['store_id'],  # Randomize at store level
)
```

---

### 2. Perturbator: Simulating Treatment Effect

The **Perturbator** simulates the treatment effect on your historical data. This lets you test "what if we had run an experiment with X% lift?"

#### Available Perturbators

| Perturbator | Effect Type | Example |
|-------------|-------------|---------|
| `ConstantPerturbator` | Absolute increase | +$5 revenue |
| `RelativePositivePerturbator` | Percentage increase | +10% revenue |
| `BinaryPerturbator` | Binary outcome shift | +5% conversion |
| `NormalPerturbator` | Normally distributed | Variable effect |

#### Example

```python
from cluster_experiments import ConstantPerturbator

perturbator = ConstantPerturbator(
    average_effect=5.0  # Add $5 to treatment group
)

# Or for relative effects
from cluster_experiments import RelativePositivePerturbator

perturbator = RelativePositivePerturbator(
    average_effect=0.10  # 10% increase
)
```

---

### 3. Analysis: Measuring Impact

The **Analysis** component specifies which statistical method to use for measuring the treatment effect.

#### Available Analysis Methods

| Analysis | Use Case |
|----------|----------|
| `OLSAnalysis` | Standard A/B test |
| `ClusteredOLSAnalysis` | Cluster randomization with clustered SE |
| `TTestClusteredAnalysis` | T-test on cluster-aggregated data |
| `GeeExperimentAnalysis` | Correlated observations (GEE) |
| `SyntheticControlAnalysis` | Observational studies |

#### Example

```python
from cluster_experiments import ClusteredOLSAnalysis

analysis = ClusteredOLSAnalysis(
    cluster_cols=['store_id'],  # Cluster standard errors
)
```

---

## Complete Example: Store-Level Experiment

Let's design a store-level promotional experiment:

```python
import pandas as pd
import numpy as np
from cluster_experiments import PowerAnalysis
from cluster_experiments import ClusteredSplitter, RelativePositivePerturbator, ClusteredOLSAnalysis

# Historical data: daily store sales
np.random.seed(42)
n_stores = 50
days = 30

historical_data = []
for store_id in range(n_stores):
    for day in range(days):
        historical_data.append({
            'store_id': store_id,
            'day': day,
            'revenue': np.random.gamma(shape=100, scale=5) + np.random.normal(0, 50)
        })

df = pd.DataFrame(historical_data)

# Define power analysis components
splitter = ClusteredSplitter(cluster_cols=['store_id'])
perturbator = RelativePositivePerturbator()  # % increase
analysis = ClusteredOLSAnalysis(cluster_cols=['store_id'])

power_analysis = PowerAnalysis(
    splitter=splitter,
    perturbator=perturbator,
    analysis=analysis,
    target_col='revenue',
    n_simulations=500
)

# Question 1: What power do we have for 10% lift?
power = power_analysis.power_analysis(df, average_effect=0.10)
print(f"Power for 10% lift: {power:.1%}")

# Question 2: How does power change with effect size?
power_curve = power_analysis.power_line(
    df, 
    average_effects=[0.05, 0.10, 0.15, 0.20]
)
print("\nPower Curve:")
print(power_curve)
```

---

## Power Curves and Timelines

### Power Curve (Effect Size)

See how power changes with different effect sizes:

```python
power_curve = power_analysis.power_line(
    df,
    average_effects=[0.03, 0.05, 0.07, 0.10, 0.15]
)
```

### MDE Timeline (Experiment Duration)

See how MDE changes with experiment length:

```python
from cluster_experiments import NormalPowerAnalysis

npw = NormalPowerAnalysis.from_dict({
    'analysis': 'clustered_ols',
    'cluster_cols': ['store_id'],
    'time_col': 'date',
})

mde_timeline = npw.mde_time_line(
    df,
    powers=[0.8],  # 80% power
    experiment_length=[7, 14, 21, 30]  # days
)
```

---

## Dictionary Configuration

For simpler setups, use dictionary configuration:

```python
from cluster_experiments import PowerAnalysis

config = {
    'splitter': 'clustered',
    'cluster_cols': ['store_id'],
    'perturbator': 'relative_positive',
    'analysis': 'clustered_ols',
    'n_simulations': 500,
}

power_analysis = PowerAnalysis.from_dict(config)
```

---

## Tips and Best Practices

### 1. Use Historical Data

- Use real historical data that matches your experiment setup
- More data = more reliable power estimates
- Ensure your historical period is representative

### 2. Match Components to Design

- If experiment is cluster-randomized, use `ClusteredSplitter` and `ClusteredOLSAnalysis`
- If individual-level, use `NonClusteredSplitter` and `OLSAnalysis`
- Match perturbator to expected effect type (absolute vs relative)

### 3. Simulation Count

- More simulations = more accurate but slower
- Start with 100-500 for exploration
- Use 1000+ for final estimates

### 4. Power Standards

- **80% power** is standard (80% chance of detecting effect if it exists)
- **Higher power** requires larger sample size or longer duration
- Consider business tradeoffs (speed vs certainty)

---

## Common Questions

### Q: What's the difference between power analysis and experiment analysis?

**Power analysis** (before experiment):
- Uses historical data
- Simulates different scenarios
- Answers: "How much data do I need?"

**Experiment analysis** (after experiment):
- Uses actual experiment data
- Measures real treatment effects
- Answers: "What was the impact?"

### Q: When should I use simulation vs normal approximation?

**Normal approximation:**
- ✅ Fast results
- ✅ Standard experimental designs
- ✅ Large sample sizes

**Simulation:**
- ✅ Complex designs (switchback, stratified)
- ✅ Non-normal distributions
- ✅ Small sample sizes

### Q: My power is too low, what can I do?

Options to increase power:
1. **Increase sample size** (more users; this can be achieved by running the experiment longer)
3. **Use variance reduction** (CUPAC/CUPED)
4. **Detect larger effects** (focus on bigger changes)
5. **Use more sensitive metrics**

---

## Next Steps

- **[Normal Power Example](normal_power.html)** - Compare simulation vs normal approximation
- **[Power Lines Example](normal_power_lines.html)** - Visualize power curves
- **[Switchback Power](switchback.html)** - Power analysis for switchback designs
- **[API Reference](api/power_analysis.html)** - Detailed power analysis documentation

