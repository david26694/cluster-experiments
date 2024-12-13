import pytest
from mktestdocs import check_docstring, check_md_file, get_codeblock_members

from ab_lab import (
    AnalysisPlan,
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    BetaRelativePerturbator,
    BetaRelativePositivePerturbator,
    BinaryPerturbator,
    ClusteredOLSAnalysis,
    ClusteredSplitter,
    ConstantPerturbator,
    ConstantWashover,
    Dimension,
    EmptyRegressor,
    EmptyWashover,
    ExperimentAnalysis,
    FixedSizeClusteredSplitter,
    GeeExperimentAnalysis,
    HypothesisTest,
    Metric,
    MLMExperimentAnalysis,
    NonClusteredSplitter,
    NormalPerturbator,
    NormalPowerAnalysis,
    OLSAnalysis,
    PairedTTestClusteredAnalysis,
    Perturbator,
    PowerAnalysis,
    PowerConfig,
    RandomSplitter,
    RatioMetric,
    RelativeMixedPerturbator,
    RelativePositivePerturbator,
    RepeatedSampler,
    SegmentedBetaRelativePerturbator,
    SimpleMetric,
    StratifiedClusteredSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
    SyntheticControlAnalysis,
    TargetAggregation,
    TTestClusteredAnalysis,
    UniformPerturbator,
    Variant,
)
from ab_lab.utils import _original_time_column

all_objects = [
    BalancedClusteredSplitter,
    BinaryPerturbator,
    ClusteredOLSAnalysis,
    ClusteredSplitter,
    EmptyRegressor,
    ExperimentAnalysis,
    FixedSizeClusteredSplitter,
    GeeExperimentAnalysis,
    NonClusteredSplitter,
    OLSAnalysis,
    Perturbator,
    PowerAnalysis,
    NormalPowerAnalysis,
    PowerConfig,
    RandomSplitter,
    StratifiedClusteredSplitter,
    SyntheticControlAnalysis,
    TargetAggregation,
    TTestClusteredAnalysis,
    PairedTTestClusteredAnalysis,
    ConstantPerturbator,
    UniformPerturbator,
    _original_time_column,
    ConstantWashover,
    EmptyWashover,
    BalancedSwitchbackSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
    RepeatedSampler,
    MLMExperimentAnalysis,
    RelativePositivePerturbator,
    NormalPerturbator,
    BetaRelativePositivePerturbator,
    BetaRelativePerturbator,
    SegmentedBetaRelativePerturbator,
    AnalysisPlan,
    Metric,
    SimpleMetric,
    RatioMetric,
    Dimension,
    Variant,
    HypothesisTest,
    RelativeMixedPerturbator,
]


def flatten(items):
    """Flattens a list"""
    return [item for sublist in items for item in sublist]


# This way we ensure that each item in `all_members` points to a method
# that could have a docstring.
all_members = flatten([get_codeblock_members(o) for o in all_objects])


@pytest.mark.parametrize("func", all_members, ids=lambda d: d.__qualname__)
def test_function_docstrings(func):
    """Test the python example in each method in each object."""
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "fpath",
    [
        "README.md",
    ],
)
def test_quickstart_docs_file(fpath):
    """Test the quickstart files."""
    check_md_file(fpath, memory=True)
