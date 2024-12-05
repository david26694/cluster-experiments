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

A library to run simulation-based power analysis, including cluster-randomized trial data. Also useful to design and analyse cluster-randomized and switchback experiments.


<img src="theme/flow.png">

## Examples

### Experiment Design
#### Hello world

Hello world of the library, non-clustered version. There is an outcome variable analyzed with a linear regression. The perturbator adds a constant effect to treated units, and the splitter is random.

```python title="Non-clustered"
import numpy as np
import pandas as pd
from cluster_experiments import PowerAnalysis

# Create fake data
N = 1_000
df = pd.DataFrame(
    {
        "target": np.random.normal(0, 1, size=N),
    }
)

config = {
    "analysis": "ols_non_clustered",
    "perturbator": "constant",
    "splitter": "non_clustered",
    "n_simulations": 50,
}
pw = PowerAnalysis.from_dict(config)

# Keep in mind that the average effect is the absolute effect added, this is not relative!
power = pw.power_analysis(df, average_effect=0.1)

# You may also get the power curve by running the power analysis with different average effects
power_line = pw.power_line(df, average_effects=[0, 0.1, 0.2])


# A faster method can be used to run the power analysis, using the approximation of
# the central limit theorem, which is stable with less simulations
from cluster_experiments import NormalPowerAnalysis
npw = NormalPowerAnalysis.from_dict(
    {
        "analysis": "ols_non_clustered",
        "splitter": "non_clustered",
        "n_simulations": 5,
    }
)
power_line_normal = npw.power_line(df, average_effects=[0, 0.1, 0.2])

# you can also use the normal power to get mde from a power level
mde = npw.mde(df, power=0.8)

```

#### Switchback

Hello world of this library, clustered version. Since it uses dates as clusters, we consider it a switchback experiment. However, if you want to run a clustered experiment, you can use the same code without the dates.

```python title="Switchback - config-based"

from datetime import date

import numpy as np
import pandas as pd
from cluster_experiments.power_analysis import PowerAnalysis

# Create fake data
N = 1_000
clusters = [f"Cluster {i}" for i in range(100)]
dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
df = pd.DataFrame(
    {
        "cluster": np.random.choice(clusters, size=N),
        "target": np.random.normal(0, 1, size=N),
        "date": np.random.choice(dates, size=N),
    }
)

config = {
    "cluster_cols": ["cluster", "date"],
    "analysis": "gee",
    "perturbator": "constant",
    "splitter": "clustered",
    "n_simulations": 50,
}
pw = PowerAnalysis.from_dict(config)

print(df)
# Keep in mind that the average effect is the absolute effect added, this is not relative!
power = pw.power_analysis(df, average_effect=0.1)
print(f"{power = }")
```

#### Long example

This is a more comprehensive example of how to use this library. There are simpler ways to run this power analysis above but this shows all the building blocks of the library.

```python title="Switchback - using classes"
from datetime import date

import numpy as np
import pandas as pd
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_analysis import PowerAnalysis, NormalPowerAnalysis
from cluster_experiments.random_splitter import ClusteredSplitter

# Create fake data
N = 1_000
clusters = [f"Cluster {i}" for i in range(100)]
dates = [f"{date(2022, 1, i):%Y-%m-%d}" for i in range(1, 32)]
df = pd.DataFrame(
    {
        "cluster": np.random.choice(clusters, size=N),
        "target": np.random.normal(0, 1, size=N),
        "date": np.random.choice(dates, size=N),
    }
)

# A switchback experiment is going to be run, prepare the switchback splitter for the analysis
sw = ClusteredSplitter(
    cluster_cols=["cluster", "date"],
)

# We use a constant perturbator to add artificial effect on the treated on the power analysis
perturbator = ConstantPerturbator()

# Use gee to run the analysis
analysis = GeeExperimentAnalysis(
    cluster_cols=["cluster", "date"],
)

# Run the power analysis
pw = PowerAnalysis(
    perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50, seed=123
)

# Keep in mind that the average effect is the absolute effect added, this is not relative!
power = pw.power_analysis(df, average_effect=0.1)
print(f"{power = }")

# You can also use normal power analysis, that uses central limit theorem to estimate power, and it should be stable in less simulations
npw = NormalPowerAnalysis(
    splitter=sw, analysis=analysis, n_simulations=50, seed=123
)
power = npw.power_analysis(df, average_effect=0.1)
print(f"{power = }")

```

### Experiment Analysis

#### Simple analysis plan

```python title="Simple Analysis Plan"

import numpy as np
import pandas as pd
from cluster_experiments import AnalysisPlan, SimpleMetric, Variant, Dimension

# ---------------------------------------
# ----------- Create fake data ----------
# ---------------------------------------

NUM_ORDERS = 10_000
NUM_CUSTOMERS = 3_000
EXPERIMENT_GROUPS = ["control", "treatment_1", "treatment_2"]
GROUP_SIZE = NUM_CUSTOMERS // len(EXPERIMENT_GROUPS)

# Generate customers and assign them to experiment groups
customer_ids = np.arange(1, NUM_CUSTOMERS + 1)
np.random.shuffle(customer_ids)
experiment_group = np.repeat(EXPERIMENT_GROUPS, GROUP_SIZE)
experiment_group = np.concatenate(
    (
        experiment_group,
        np.random.choice(EXPERIMENT_GROUPS, NUM_CUSTOMERS - len(experiment_group)),
    )
)
customer_group_mapping = dict(zip(customer_ids, experiment_group))

# Generate orders
order_ids = np.arange(1, NUM_ORDERS + 1)
customers = np.random.choice(customer_ids, NUM_ORDERS)
order_values = np.abs(
    np.random.normal(loc=10, scale=2, size=NUM_ORDERS)
)  # Normally distributed around 10 and positive
order_delivery_times = np.abs(
    np.random.normal(loc=30, scale=5, size=NUM_ORDERS)
)  # Normally distributed around 30 minutes and positive
order_city_codes = np.random.randint(
    1, 3, NUM_ORDERS
)  # Random city codes between 1 and 2

# Create DataFrame
data = {
    "order_id": order_ids,
    "customer_id": customers,
    "experiment_group": [
        customer_group_mapping[customer_id] for customer_id in customers
    ],
    "order_value": order_values,
    "order_delivery_time": order_delivery_times,
    "order_city_code": order_city_codes,
}

df = pd.DataFrame(data)
df.order_city_code = df.order_city_code.astype(str)

# -------------------------------------------------
# ---- Define metrics, variants and dimensions ----
# -------------------------------------------------

dimension__city_code = Dimension(name="order_city_code", values=["1", "2"])

metric__order_value = SimpleMetric(alias="AOV", name="order_value")

metric__delivery_time = SimpleMetric(
    alias="AVG DT", name="order_delivery_time_in_minutes"
)

variants = [
    Variant("control", is_control=True),
    Variant("treatment_1", is_control=False),
    Variant("treatment_2", is_control=False),
]

# --------------------------------------------------------
# ---- Define a simple analysis plan from the metrics ----
# --------------------------------------------------------

simple_analysis_plan = AnalysisPlan.from_metrics(
    metrics=[metric__delivery_time, metric__order_value],
    variants=variants,
    variant_col="experiment_group",
    alpha=0.01,
    dimensions=[dimension__city_code],
    analysis_type="clustered_ols",
    analysis_config={"cluster_cols": ["customer_id"]},
)

# --------------------------------------------------------
# ---- Run analysis and get the experiment scorecard  ----
# --------------------------------------------------------

simple_results = simple_analysis_plan.analyze(exp_data=df, verbose=True)

simple_results_df = simple_results.to_dataframe()
```

## Features

The library offers the following classes:

* Regarding power analysis:
    * `PowerAnalysis`: to run power analysis on any experiment design, using simulation
    * `PowerAnalysisWithPreExperimentData`: to run power analysis on a clustered/switchback design, but adding pre-experiment df during split and perturbation (especially useful for Synthetic Control)
    * `NormalPowerAnalysis`: to run power analysis on any experiment design using the central limit theorem for the distribution of the estimator. It can be used to compute the minimum detectable effect (MDE) for a given power level.
    * `ConstantPerturbator`: to artificially perturb treated group with constant perturbations
    * `BinaryPerturbator`: to artificially perturb treated group for binary outcomes
    * `RelativePositivePerturbator`: to artificially perturb treated group with relative positive perturbations
    * `RelativeMixedPerturbator`: to artificially perturb treated group with relative perturbations for positive and negative targets
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

## Installation

You can install this package via `pip`.

```bash
pip install cluster-experiments
```

It may be safer to install via;

```bash
python -m pip install cluster-experiments
```

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
