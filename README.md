<img src="theme/icon-cluster.png" width=200 height=200 align="right">

# cluster_experiments

[![Downloads](https://static.pepy.tech/badge/cluster-experiments)](https://pepy.tech/project/cluster-experiments)
[![PyPI](https://img.shields.io/pypi/v/cluster-experiments)](
https://pypi.org/project/cluster-experiments/)
[![Unit tests](https://github.com/david26694/cluster-experiments/workflows/Release%20unit%20Tests/badge.svg)](https://github.com/david26694/cluster-experiments/actions)
[![CodeCov](
https://codecov.io/gh/david26694/cluster-experiments/branch/main/graph/badge.svg)](https://app.codecov.io/gh/david26694/cluster-experiments/)
![License](https://img.shields.io/github/license/david26694/cluster-experiments)
[![Pypi version](https://img.shields.io/pypi/pyversions/cluster-experiments.svg)](https://pypi.python.org/pypi/cluster-experiments)

A Python library for end-to-end A/B testing workflows, featuring:
- Experiment analysis and scorecards
- Power analysis (simulation-based and normal approximation)
- Variance reduction techniques (CUPED, CUPAC)
- Support for complex experimental designs (cluster randomization, switchback experiments)

<img src="theme/flow.png">

## Key Features

### 1. Power Analysis
- **Simulation-based**: Run Monte Carlo simulations to estimate power
- **Normal approximation**: Fast power estimation using CLT
- **Minimum Detectable Effect**: Calculate required effect sizes
- **Multiple designs**: Support for:
  - Simple randomization
  - Variance reduction techniques in power analysis
  - Cluster randomization
  - Switchback experiments
- **Dict config**: Easy to configure power analysis with a dictionary

### 2. Experiment Analysis
- **Analysis Plans**: Define structured analysis plans
- **Metrics**:
  - Simple metrics
  - Ratio metrics
- **Dimensions**: Slice results by dimensions
- **Statistical Methods**:
  - GEE
  - Mixed Linear Models
  - Clustered / regular OLS
  - T-tests
  - Synthetic Control
- **Dict config**: Easy to define analysis plans with a dictionary

### 3. Variance Reduction
- **CUPED** (Controlled-experiment Using Pre-Experiment Data):
  - Use historical outcome data to reduce variance, choose any granularity
  - Support for several covariates
- **CUPAC** (Control Using Predictors as Covariates):
  - Use any scikit-learn compatible estimator to predict the outcome with pre-experiment data

## Quick Start

### Power Analysis Example

```python
import numpy as np
import pandas as pd
from cluster_experiments import PowerAnalysis, NormalPowerAnalysis

# Create sample data
N = 1_000
df = pd.DataFrame({
    "target": np.random.normal(0, 1, size=N),
    "date": pd.to_datetime(
        np.random.randint(
            pd.Timestamp("2024-01-01").value,
            pd.Timestamp("2024-01-31").value,
            size=N,
        )
    ),
})

# Simulation-based power analysis with CUPED
config = {
    "analysis": "ols",
    "perturbator": "constant",
    "splitter": "non_clustered",
    "n_simulations": 50,
}
pw = PowerAnalysis.from_dict(config)
power = pw.power_analysis(df, average_effect=0.1)

# Normal approximation (faster)
npw = NormalPowerAnalysis.from_dict({
    "analysis": "ols",
    "splitter": "non_clustered",
    "n_simulations": 5,
    "time_col": "date",
})
power_normal = npw.power_analysis(df, average_effect=0.1)
power_line_normal = npw.power_line(df, average_effects=[0.1, 0.2, 0.3])


# MDE calculation
mde = npw.mde(df, power=0.8)

# MDE line with length
mde_timeline = npw.mde_time_line(
    df,
    powers=[0.8],
    experiment_length=[7, 14, 21]
)

print(power, power_line_normal, power_normal, mde, mde_timeline)
```

### Experiment Analysis Example

```python
import numpy as np
import pandas as pd
from cluster_experiments import AnalysisPlan, SimpleMetric, Variant, Dimension

N = 1_000
experiment_data = pd.DataFrame({
    "order_value": np.random.normal(100, 10, size=N),
    "delivery_time": np.random.normal(10, 1, size=N),
    "experiment_group": np.random.choice(["control", "treatment"], size=N),
    "city": np.random.choice(["NYC", "LA"], size=N),
    "customer_id": np.random.randint(1, 100, size=N),
    "customer_age": np.random.randint(20, 60, size=N),
})

# Define metrics
aov = SimpleMetric(alias="AOV", name="order_value")
delivery_time = SimpleMetric(alias="Delivery Time", name="delivery_time")

# Define variants and dimensions
variants = [
    Variant("control", is_control=True),
    Variant("treatment", is_control=False),
]
city_dimension = Dimension(name="city", values=["NYC", "LA"])

# Create analysis plan with CUPAC
plan = AnalysisPlan.from_metrics(
    metrics=[aov, delivery_time],
    variants=variants,
    variant_col="experiment_group",
    dimensions=[city_dimension],
    analysis_type="clustered_ols",
    analysis_config={
        "cluster_cols": ["customer_id"],
    },
)

# Run analysis
results = plan.analyze(experiment_data)
scorecard = results.to_dataframe()
print(scorecard)
```

### Variance Reduction Example

```python
import numpy as np
import pandas as pd
from cluster_experiments import (
    AnalysisPlan,
    SimpleMetric,
    Variant,
    Dimension,
    TargetAggregation,
    HypothesisTest
)

N = 1000

experiment_data = pd.DataFrame({
    "order_value": np.random.normal(100, 10, size=N),
    "delivery_time": np.random.normal(10, 1, size=N),
    "experiment_group": np.random.choice(["control", "treatment"], size=N),
    "city": np.random.choice(["NYC", "LA"], size=N),
    "customer_id": np.random.randint(1, 100, size=N),
    "customer_age": np.random.randint(20, 60, size=N),
})

pre_experiment_data = pd.DataFrame({
    "order_value": np.random.normal(100, 10, size=N),
    "customer_id": np.random.randint(1, 100, size=N),
})

# Define metrics
aov = SimpleMetric(alias="AOV", name="order_value")
delivery_time = SimpleMetric(alias="Delivery Time", name="delivery_time")

# Define variants and dimensions
variants = [
    Variant("control", is_control=True),
    Variant("treatment", is_control=False),
]
city_dimension = Dimension(name="city", values=["NYC", "LA"])

cupac_model = TargetAggregation(
    agg_col="customer_id",
    target_col="order_value"
)

hypothesis_test = HypothesisTest(
    metric=aov,
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

# Create analysis plan with CUPAC
plan = AnalysisPlan(
    tests=[hypothesis_test],
    variants=variants,
    variant_col="experiment_group",
)

# Run analysis
results = plan.analyze(experiment_data, pre_experiment_data)
scorecard = results.to_dataframe()
print(scorecard)
```

## Installation

You can install this package via `pip`.

```bash
pip install cluster-experiments
```

For detailed documentation and examples, visit our [documentation site](https://cluster-experiments.readthedocs.io/en/latest/).

## Features

The library offers the following classes:

* Regarding power analysis:
    * `PowerAnalysis`: to run power analysis on any experiment design, using simulation
    * `PowerAnalysisWithPreExperimentData`: to run power analysis on a clustered/switchback design, but adding pre-experiment df during split and perturbation (especially useful for Synthetic Control)
    * `NormalPowerAnalysis`: to run power analysis on any experiment design using the central limit theorem for the distribution of the estimator. It can be used to compute the minimum detectable effect (MDE) for a given power level.
    * `ConstantPerturbator`: to artificially perturb treated group with constant perturbations
    * `BinaryPerturbator`: to artificially perturb treated group for binary outcomes
    * `RelativePositivePerturbator`: to artificially perturb treated group with relative positive perturbations
    * `NormalPerturbator`: to artificially perturb treated group with normal distribution perturbations
    * `BetaRelativePositivePerturbator`: to artificially perturb treated group with relative positive beta distribution perturbations
    * `BetaRelativePerturbator`: to artificially perturb treated group with relative beta distribution perturbations in a specified support interval
    * `SegmentedBetaRelativePerturbator`: to artificially perturb treated group with relative beta distribution perturbations in a specified support interval, but using clusters
* Regarding splitting data:
    * `ClusteredSplitter`: to split data based on clusters
    * `FixedSizeClusteredSplitter`: to split data based on clusters with a fixed size (example: only 1 treatment cluster and the rest in control)
    * `BalancedClusteredSplitter`: to split data based on clusters in a balanced way
    * `NonClusteredSplitter`: Regular data splitting, no clusters
    * `StratifiedClusteredSplitter`: to split based on clusters and strata, balancing the number of clusters in each stratus
    * `RepeatedSampler`: for backtests where we have access to counterfactuals, does not split the data, just duplicates the data for all groups
    * Switchback splitters (the same can be done with clustered splitters, but there is a convenient way to define switchback splitters using switch frequency):
        * `SwitchbackSplitter`: to split data based on clusters and dates, for switchback experiments
        * `BalancedSwitchbackSplitter`: to split data based on clusters and dates, for switchback experiments, balancing treatment and control among all clusters
        * `StratifiedSwitchbackSplitter`: to split data based on clusters and dates, for switchback experiments, balancing the number of clusters in each stratus
        * Washover for switchback experiments:
            * `EmptyWashover`: no washover done at all.
            * `ConstantWashover`: accepts a timedelta parameter and removes the data when we switch from A to B for the timedelta interval.
* Regarding analysis methods:
    * `GeeExperimentAnalysis`: to run GEE analysis on the results of a clustered design
    * `MLMExperimentAnalysis`: to run Mixed Linear Model analysis on the results of a clustered design
    * `TTestClusteredAnalysis`: to run a t-test on aggregated data for clusters
    * `PairedTTestClusteredAnalysis`: to run a paired t-test on aggregated data for clusters
    * `ClusteredOLSAnalysis`: to run OLS analysis on the results of a clustered design
    * `OLSAnalysis`: to run OLS analysis for non-clustered data
    * `TargetAggregation`: to add pre-experimental data of the outcome to reduce variance
    * `SyntheticControlAnalysis`: to run synthetic control analysis
* Regarding experiment analysis workflow:
    * `Metric`: abstract class to define a metric to be used in the analysis
    * `SimpleMetric`: to create a metric defined at the same level of the data used for the analysis
    * `RatioMetric`: to create a metric defined at a lower level than the data used for the analysis
    * `Variant`: to define a variant of the experiment
    * `Dimension`: to define a dimension to slice the results of the experiment
    * `HypothesisTest`: to define a Hypothesis Test with a metric, analysis method, optional analysis configuration, and optional dimensions
    * `AnalysisPlan`: to define a plan of analysis with a list of Hypothesis Tests for a dataset and the experiment variants. The `analyze()` method runs the analysis and returns the results
    * `AnalysisResults`: to store the results of the analysis
* Other:
    * `PowerConfig`: to conveniently configure `PowerAnalysis` class
    * `ConfidenceInterval`: to store the data representation of a confidence interval
    * `InferenceResults`: to store the structure of complete statistical analysis results



## Contributing

In case you want to use venv as a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

After creating the virtual environment (or not), run:
```bash
git clone git@github.com:david26694/cluster-experiments.git
cd cluster-experiments
make install-dev
```
