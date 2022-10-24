<img src="theme/icon-cluster.png" width=200 height=200 align="right">


# cluster_experiments

A library to run simulation-based power analysis, including clustered data. Also useful to design and analyse clustered and switchback experiments.

## Examples

### Base example

Hello world of this library:

```python

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
    "perturbator": "uniform",
    "splitter": "clustered",
    "n_simulations": 50,
}
pw = PowerAnalysis.from_dict(config)

print(df)
power = pw.power_analysis(df, average_effect=0.1)
print(f"{power = }")

```

### Long example

This is a comprehensive example of how to use this library. There are simpler ways to run power analysis but this shows all the building blocks of the library.

```python
from datetime import date

import numpy as np
import pandas as pd
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
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

# We use a uniform perturbator to add artificial effect on the treated on the power analysis
perturbator = UniformPerturbator()

# Use gee to run the analysis
analysis = GeeExperimentAnalysis(
    cluster_cols=["cluster", "date"],
)

# Run the power analysis
pw = PowerAnalysis(
    perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
)

power = pw.power_analysis(df, average_effect=0.1)
print(f"{power = }")
```

## Features

The library offers the following classes:

* Regarding power analysis:
    * `PowerAnalysis`: to run power analysis on a clustered/switchback design
    * `UniformPerturbator`: to artificially perturb treated group with uniform perturbations
    * `BinaryPerturbator`: to artificially perturb treated group for binary outcomes
* Regarding splitting data:
    * `ClusteredSplitter`: to split data based on clusters
    * `BalancedClusteredSplitter`: to split data based on clusters in a balanced way
    * `NonClusteredSplitter`: Regular data splitting, no clusters
    * `StratifiedClusteredSplitter`: to split based on clusters and strata, balancing the number of clusters in each stratus
* Regarding analysis:
    * `GeeExperimentAnalysis`: to run GEE analysis on the results of a clustered design
    * `TTestClusteredAnalysis`: to run a t-test on aggregated data for clusters
    * `ClusteredOLSAnalysis`: to run OLS analysis on the results of a clustered design
    * `OLSAnalysis`: to run OLS analysis for non-clustered data
    * `TargetAggregation`: to add pre-experimental data of the outcome to reduce variance
* Other:
    * `PowerConfig`: to conviently configure `PowerAnalysis` class

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
