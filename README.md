# cluster_experiments

A library to run power analysis and analyse clustered and switchback experiments.

## Example

This is a comprehensive example of how to use this library. There are simpler ways to run power analysis but this shows all the building blocks of the library.

```python
from datetime import date

import numpy as np
import pandas as pd
from cluster_experiments.experiment_analysis import GeeExperimentAnalysis
from cluster_experiments.perturbator import UniformPerturbator
from cluster_experiments.power_analysis import PowerAnalysis
from cluster_experiments.random_splitter import SwitchbackSplitter

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
sw = SwitchbackSplitter(
    clusters=clusters,
    dates=dates,
)

# We use a uniform perturbator to add artificial effect on the treated on the power analysis
perturbator = UniformPerturbator(
    average_effect=0.1,
)

# Use gee to run the analysis
analysis = GeeExperimentAnalysis(
    cluster_cols=["cluster", "date"],
)

# Run the power analysis
pw = PowerAnalysis(
    perturbator=perturbator, splitter=sw, analysis=analysis, n_simulations=50
)

power = pw.power_analysis(df)
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
    * `SwitchbackSplitter`: to split data based using switchback method
    * `BalancedClusteredSplitter`: to split data based on clusters in a balanced way
    * `BalancedSwitchbackSplitter`: to split data based using switchback method in a balanced way
* Regarding analysis:
    * `GeeExperimentAnalysis`: to run GEE analysis on a the results of a clustered design
    * `TargetAggregation`: to add pre-experimental data of the outcome to reduce variance
* Other:
    * `PowerConfig`: to conviently configure `PowerAnalysis` class

## Installation

You can install this package via `pip`.

```
pip install cluster-experiments
```

It may be safer to install via;

```
python -m pip install cluster-experiments
```

## Contributing

```
git clone git@github.com:david26694/cluster-experiments.git
cd cluster-experiments
make install-dev
```
