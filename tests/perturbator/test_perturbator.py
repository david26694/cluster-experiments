import pandas as pd
import pytest

from cluster_experiments.perturbator import BinaryPerturbator, UniformPerturbator
from tests.examples import binary_df, continuous_df


@pytest.mark.parametrize("average_effect, output_value", [(-1, 0), (1, 1)])
def test_binary_perturbator_all_1(average_effect, output_value):
    bp = BinaryPerturbator(average_effect=average_effect)

    assert (
        bp.perturbate(binary_df).query("treatment == 'B'")["target"] == output_value
    ).all()


@pytest.mark.parametrize(
    "average_effect, output_values", [(-0.1, [60, 40]), (0.1, [40, 60])]
)
def test_binary_perturbator_10(average_effect, output_values):
    binary_df_repeated = pd.concat([binary_df for _ in range(50)])
    bp = BinaryPerturbator()
    assert (
        bp.perturbate(binary_df_repeated, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .value_counts()
        .loc[[0, 1]]
        == output_values
    ).all()


@pytest.mark.parametrize(
    "average_effect, output_values", [(-0.1, [60, 40]), (0.1, [40, 60])]
)
def test_binary_perturbator_10_perturbate(average_effect, output_values):
    binary_df_repeated = pd.concat([binary_df for _ in range(50)])
    bp = BinaryPerturbator(average_effect=average_effect)
    assert (
        bp.perturbate(binary_df_repeated)
        .query("treatment == 'B'")["target"]
        .value_counts()
        .loc[[0, 1]]
        == output_values
    ).all()


@pytest.mark.parametrize("average_effect, avg_target", [(-0.1, 0.4), (0.1, 0.6)])
def test_uniform_perturbator(average_effect, avg_target):
    up = UniformPerturbator(average_effect=average_effect)
    assert (
        up.perturbate(continuous_df).query("treatment == 'B'")["target"].mean()
        == avg_target
    )


@pytest.mark.parametrize("average_effect, avg_target", [(-0.1, 0.4), (0.1, 0.6)])
def test_uniform_perturbator_perturbate(average_effect, avg_target):
    up = UniformPerturbator()
    assert (
        up.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .mean()
        == avg_target
    )


def test_binary_raises():
    binary_df_repeated = pd.concat([binary_df for _ in range(50)])
    bp = BinaryPerturbator()
    with pytest.raises(ValueError, match="average_effect must be provided"):
        bp.perturbate(binary_df_repeated)


@pytest.mark.parametrize("average_effect", [(-1.1), (1.1)])
def test_binary_raises_out_of_limit(average_effect):
    bp = BinaryPerturbator()
    with pytest.raises(ValueError, match="Average effect must be in"):
        bp.perturbate(binary_df, average_effect=average_effect)


def test_binary_raises_non_binary_target():
    bp = BinaryPerturbator()
    binary_df["target"] = binary_df["target"] + 0.01
    with pytest.raises(ValueError, match="must be binary"):
        bp.perturbate(binary_df, average_effect=0.05)
