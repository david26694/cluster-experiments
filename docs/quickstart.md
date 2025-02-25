# Quickstart

## Installation

You can install **Cluster Experiments** via pip:

```bash
pip install cluster-experiments
```

!!! info "Python Version Support"
    **Cluster Experiments** requires **Python 3.9 or higher**. Make sure your environment meets this requirement before proceeding with the installation.

---

## Usage

Designing and analyzing experiments can feel overwhelming at times. After formulating a testable hypothesis,
you're faced with a series of routine tasks. From collecting and transforming raw data to measuring the statistical significance of your experiment results and constructing confidence intervals,
it can quickly become a repetitive and error-prone process.
*Cluster Experiments* is here to change that. Built on top of well-known packages like `pandas`, `numpy`, `scipy` and `statsmodels`,  it automates the core steps of an experiment, streamlining your workflow, saving you time and effort, while maintaining statistical rigor.
## Key Features
- **Modular Design**: Each component—`Splitter`, `Perturbator`, and `Analysis`—is independent, reusable, and can be combined in any way you need.
- **Flexibility**: Whether you're conducting a simple A/B test or a complex clustered experiment, Cluster Experiments adapts to your needs.
- **Statistical Rigor**: Built-in support for advanced statistical methods ensures that your experiments maintain high standards, including clustered standard errors and variance reduction techniques like CUPED and CUPAC.

The core functionality of *Cluster Experiments* revolves around several intuitive, self-contained classes and methods:

- **Splitter**: Define how your control and treatment groups are split.
- **Perturbator**: Specify the type of effect you want to test.
- **Analysis**: Perform statistical inference to measure the impact of your experiment.


---

### `Splitter`: Defining Control and Treatment Groups

The `Splitter` classes are responsible for dividing your data into control and treatment groups. The way you split your data depends on the **metric** (e.g., simple, ratio) you want to observe and the unit of observation (e.g., users, sessions, time periods).

#### Features:

- **Randomized Splits**: Simple random assignment of units to control and treatment groups.
- **Stratified Splits**: Ensure balanced representation of key segments (e.g., geographic regions, user cohorts).
- **Time-Based Splits**: Useful for switchback experiments or time-series data.

```python
from cluster_experiments import RandomSplitter

splitter = RandomSplitter(
    cluster_cols=["cluster_id"],  # Split by clusters
    treatment_col="treatment",    # Name of the treatment column
)
```

---

### `Perturbator`: Simulating the Treatment Effect

The `Perturbator` classes define the type of effect you want to test. It simulates the treatment effect on your data, allowing you to evaluate the impact of your experiment.

#### Features:

- **Absolute Effects**: Add a fixed uplift to the treatment group.
- **Relative Effects**: Apply a percentage-based uplift to the treatment group.
- **Custom Effects**: Define your own effect size or distribution.

```python
from cluster_experiments import ConstantPerturbator

perturbator = ConstantPerturbator(
    average_effect=5.0  # Simulate a nominal 5% uplift 
)
```

---

### `Analysis`: Measuring the Impact

Once your data is split and the treatment effect is applied, the `Analysis` component helps you measure the statistical significance of the experiment results. It provides tools for calculating effects, confidence intervals, and p-values.

You can use it for both **experiment design** (pre-experiment phase) and **analysis** (post-experiment phase).

#### Features:

- **Statistical Tests**: Perform t-tests, OLS regression, and other hypothesis tests.
- **Effect Size**: Calculate both absolute and relative effects.
- **Confidence Intervals**: Construct confidence intervals for your results.

Example:

```python
from cluster_experiments import TTestClusteredAnalysis

analysis = TTestClusteredAnalysis(
    cluster_cols=["cluster_id"],  # Cluster-level analysis
    treatment_col="treatment",    # Name of the treatment column
    target_col="outcome"          # Metric to analyze
)
```

---

### Putting It All Together for Experiment Design

You can combine all classes as inputs in the `PowerAnalysis` class, where you can analyze different experiment settings, power lines, and Minimal Detectable Effects (MDEs).

```python
from cluster_experiments import PowerAnalysis
from cluster_experiments import RandomSplitter, ConstantPerturbator, TTestClusteredAnalysis

# Define the components
splitter = RandomSplitter(cluster_cols=["cluster_id"], treatment_col="treatment")
perturbator = ConstantPerturbator(average_effect=0.1)
analysis = TTestClusteredAnalysis(cluster_cols=["cluster_id"], treatment_col="treatment", target_col="outcome")

# Create the experiment
experiment = PowerAnalysis(
    perturbator=perturbator,
    splitter=splitter,
    analysis=analysis,
    target_col="outcome",
    treatment_col="treatment"
)

# Run the experiment
results = experiment.power_analysis()
```

---

## Next Steps

- Explore the **Core Documentation** for detailed explanations of each component.
- Check out the **Usage Examples** for practical applications of the package.
