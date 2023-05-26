import pytest
from mktestdocs import check_docstring, check_md_file, get_codeblock_members

from cluster_experiments import (
    BalancedClusteredSplitter,
    BalancedSwitchbackSplitter,
    BetaRelativePerturbator,
    BetaRelativePositivePerturbator,
    BinaryPerturbator,
    ClusteredOLSAnalysis,
    ClusteredSplitter,
    ConstantPerturbator,
    ConstantWashover,
    EmptyRegressor,
    EmptyWashover,
    ExperimentAnalysis,
    GeeExperimentAnalysis,
    MLMExperimentAnalysis,
    NonClusteredSplitter,
    NormalPerturbator,
    OLSAnalysis,
    PairedTTestClusteredAnalysis,
    Perturbator,
    PowerAnalysis,
    PowerConfig,
    RandomSplitter,
    RelativePositivePerturbator,
    RepeatedSampler,
    SegmentedBetaRelativePerturbator,
    StratifiedClusteredSplitter,
    StratifiedSwitchbackSplitter,
    SwitchbackSplitter,
    TargetAggregation,
    TTestClusteredAnalysis,
    UniformPerturbator,
)
from cluster_experiments.utils import _original_time_column

all_objects = [
    BalancedClusteredSplitter,
    BinaryPerturbator,
    ClusteredOLSAnalysis,
    ClusteredSplitter,
    EmptyRegressor,
    ExperimentAnalysis,
    GeeExperimentAnalysis,
    NonClusteredSplitter,
    OLSAnalysis,
    Perturbator,
    PowerAnalysis,
    PowerConfig,
    RandomSplitter,
    StratifiedClusteredSplitter,
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
