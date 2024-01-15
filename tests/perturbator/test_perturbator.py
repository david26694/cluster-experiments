import numpy as np
import pandas as pd
import pytest

from cluster_experiments.perturbator import (
    BetaRelativePerturbator,
    BetaRelativePositivePerturbator,
    BinaryPerturbator,
    ConstantPerturbator,
    NormalPerturbator,
    RelativePositivePerturbator,
    SegmentedBetaRelativePerturbator,
    UniformPerturbator,
)
from cluster_experiments.power_config import PowerConfig


def test_binary_perturbator_from_config():
    config = PowerConfig(
        analysis="gee",
        splitter="clustered_balance",
        perturbator="binary",
        cluster_cols=["cluster"],
        n_simulations=100,
    )
    bp = BinaryPerturbator.from_config(config)
    assert isinstance(bp, BinaryPerturbator)


@pytest.mark.parametrize("average_effect, output_value", [(-1, 0), (1, 1)])
def test_binary_perturbator_all_1(average_effect, output_value, binary_df):
    bp = BinaryPerturbator(average_effect=average_effect)

    assert (
        bp.perturbate(binary_df).query("treatment == 'B'")["target"] == output_value
    ).all()


@pytest.mark.parametrize(
    "average_effect, output_values", [(-0.1, [60, 40]), (0.1, [40, 60])]
)
def test_binary_perturbator_10(average_effect, output_values, binary_df):
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
def test_binary_perturbator_10_perturbate(average_effect, output_values, binary_df):
    binary_df_repeated = pd.concat([binary_df for _ in range(50)])
    bp = BinaryPerturbator(average_effect=average_effect)
    assert (
        bp.perturbate(binary_df_repeated)
        .query("treatment == 'B'")["target"]
        .value_counts()
        .loc[[0, 1]]
        == output_values
    ).all()


@pytest.mark.parametrize(
    "average_effect, avg_target, perturbator",
    [
        (-0.1, 0.4, ConstantPerturbator),
        (0.1, 0.6, ConstantPerturbator),
        (-0.1, 0.4, UniformPerturbator),
        (0.1, 0.6, UniformPerturbator),
    ],
)
def test_constant_perturbator(average_effect, avg_target, perturbator, continuous_df):
    up = perturbator(average_effect=average_effect)
    assert (
        up.perturbate(continuous_df).query("treatment == 'B'")["target"].mean()
        == avg_target
    )


@pytest.mark.parametrize(
    "average_effect, avg_target, perturbator",
    [
        (-0.1, 0.4, ConstantPerturbator),
        (0.1, 0.6, ConstantPerturbator),
        (-0.1, 0.4, UniformPerturbator),
        (0.1, 0.6, UniformPerturbator),
    ],
)
def test_constant_perturbator_perturbate(
    average_effect, avg_target, perturbator, continuous_df
):
    up = perturbator()
    assert (
        up.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .mean()
        == avg_target
    )


@pytest.mark.parametrize("average_effect", [-0.1, 0.1])
def test_normal_perturbator_perturbate(average_effect, continuous_df):
    # given
    np.random.seed(24)
    effect = (
        np.random.normal(average_effect, abs(average_effect), 2)
        + continuous_df.query("treatment == 'B'")["target"].values
    )

    # when
    np.random.seed(24)
    stochastic_perturbator = NormalPerturbator()
    perturbated_values = (
        stochastic_perturbator.perturbate(continuous_df, average_effect)
        .query("treatment == 'B'")["target"]
        .values
    )

    # then
    assert np.mean(perturbated_values) == np.mean(effect)
    assert np.var(perturbated_values) == np.var(effect)


@pytest.mark.parametrize("average_effect, scale", [(-0.1, 0.02), (0.1, 0.03)])
def test_normal_scale_provided_is_used(average_effect, scale, continuous_df):
    # given
    np.random.seed(24)
    effect = (
        np.random.normal(average_effect, scale, 2)
        + continuous_df.query("treatment == 'B'")["target"].values
    )

    # when
    np.random.seed(24)
    stochastic_perturbator = NormalPerturbator(scale=scale)
    perturbated_values = (
        stochastic_perturbator.perturbate(continuous_df, average_effect)
        .query("treatment == 'B'")["target"]
        .values
    )

    # then
    assert np.mean(perturbated_values) == np.mean(effect)
    assert np.var(perturbated_values) == np.var(effect)


def test_normal_perturbator_from_config():
    config = PowerConfig(
        analysis="gee",
        splitter="clustered_balance",
        perturbator="normal",
        cluster_cols=["cluster"],
        n_simulations=100,
    )
    np = NormalPerturbator.from_config(config)
    assert isinstance(np, NormalPerturbator)


@pytest.mark.parametrize("average_effect, avg_target", [(0.1, 0.55), (0.04, 0.52)])
def test_relative_positive_perturbate(average_effect, avg_target, continuous_df):
    rp = RelativePositivePerturbator()
    assert (
        rp.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .mean()
        == avg_target
    )


@pytest.mark.parametrize("average_effect", [0.1, 0.04])
def test_stochastic_relative_perturbate(average_effect, continuous_df):
    # given
    rp = BetaRelativePositivePerturbator()
    mean = average_effect / (average_effect * average_effect)
    variance = (1 - average_effect) / (average_effect * average_effect)
    np.random.seed(24)
    effect = (1 + np.random.beta(mean, variance, 2)) * continuous_df.query(
        "treatment == 'B'"
    )["target"].values

    # when
    np.random.seed(24)
    perturbated_values = (
        rp.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .values
    )

    # then
    assert np.mean(perturbated_values) == np.mean(effect)
    assert np.var(perturbated_values) == np.var(effect)


@pytest.mark.parametrize("average_effect, scale", [(0.2, 0.05), (0.4, 0.1)])
def test_stochastic_relative_perturbate_scale_provided_is_used(
    average_effect, scale, continuous_df
):
    # given
    rp = BetaRelativePositivePerturbator(scale=scale)
    mean = average_effect / (scale * scale)
    variance = (1 - average_effect) / (scale * scale)
    np.random.seed(24)
    effect = (1 + np.random.beta(mean, variance, 2)) * continuous_df.query(
        "treatment == 'B'"
    )["target"].values

    # when
    np.random.seed(24)
    perturbated_values = (
        rp.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .values
    )

    # then
    assert np.mean(perturbated_values) == np.mean(effect)
    assert np.var(perturbated_values) == np.var(effect)


@pytest.mark.parametrize("average_effect", [0.1, 0.04])
def test_beta_relative_perturbate(average_effect, continuous_df):
    # given
    range_min = -0.8
    range_max = 5
    pert = BetaRelativePerturbator(range_min=range_min, range_max=range_max)

    average_effect_inv_transf = (average_effect - range_min) / (range_max - range_min)
    scale_inv_transf = average_effect_inv_transf
    a = average_effect_inv_transf / (scale_inv_transf * scale_inv_transf)
    b = (1 - average_effect_inv_transf) / (scale_inv_transf * scale_inv_transf)

    np.random.seed(24)
    beta = np.random.beta(a / abs(average_effect), b / abs(average_effect), 2)
    beta_transf = beta * (range_max - range_min) + range_min

    effect = (1 + beta_transf) * continuous_df.query("treatment == 'B'")[
        "target"
    ].values

    # when
    np.random.seed(24)
    perturbated_values = (
        pert.perturbate(continuous_df, average_effect=average_effect)
        .query("treatment == 'B'")["target"]
        .values
    )

    # then
    assert np.isclose(np.mean(perturbated_values), np.mean(effect))
    assert np.isclose(np.var(perturbated_values), np.var(effect))


@pytest.mark.parametrize("average_effect", [0.1, 0.04])
def test_segmented_beta_relative_perturbate(average_effect, generate_clustered_data):
    range_min = -0.8
    range_max = 4

    df_clustered = generate_clustered_data
    pert = SegmentedBetaRelativePerturbator(
        range_min=range_min, range_max=range_max, segment_cols=["city_code"]
    )

    mean_effects = []
    # repeat multiple times to decrease variability
    for _ in range(100):
        df_pert = pert.perturbate(
            pd.concat([df_clustered for _ in range(100)]), average_effect=average_effect
        )
        df_merged = df_clustered.merge(
            df_pert,
            on=["country_code", "city_code", "user_id", "date", "treatment"],
            suffixes=["", "_pert"],
        )
        df_merged = df_merged.assign(
            rel_pert=df_merged["target_pert"] / df_merged["target"]
        )
        mean_effects.append(
            df_merged.loc[df_merged["treatment"] == "B", "rel_pert"].mean()
        )

    # maximum relative difference we expect between the simulated relative perturbations and the passed average_effect
    max_rel_diff = 0.2
    assert abs((np.mean(mean_effects) - 1) / average_effect - 1) < max_rel_diff


@pytest.mark.parametrize("average_effect", [0.1, 0.04])
def test_segmented_beta_relative_perturbate_multiple_segments(
    average_effect, generate_clustered_data
):
    range_min = -0.8
    range_max = 4

    df_clustered = generate_clustered_data
    pert = SegmentedBetaRelativePerturbator(
        range_min=range_min, range_max=range_max, segment_cols=["city_code", "date"]
    )

    mean_effects = []
    # repeat multiple times to decrease variability
    for _ in range(100):
        df_pert = pert.perturbate(
            pd.concat([df_clustered for _ in range(100)]), average_effect=average_effect
        )
        df_merged = df_clustered.merge(
            df_pert,
            on=["country_code", "city_code", "user_id", "date", "treatment"],
            suffixes=["", "_pert"],
        )
        df_merged = df_merged.assign(
            rel_pert=df_merged["target_pert"] / df_merged["target"]
        )
        mean_effects.append(
            df_merged.loc[df_merged["treatment"] == "B", "rel_pert"].mean()
        )

    # maximum relative difference we expect between the simulated relative perturbations and the passed average_effect
    max_rel_diff = 0.2
    assert abs((np.mean(mean_effects) - 1) / average_effect - 1) < max_rel_diff


def test_binary_raises(binary_df):
    binary_df_repeated = pd.concat([binary_df for _ in range(50)])
    bp = BinaryPerturbator()
    with pytest.raises(ValueError, match="average_effect must be provided"):
        bp.perturbate(binary_df_repeated)


@pytest.mark.parametrize("average_effect", [(-1.1), (1.1)])
def test_binary_raises_out_of_limit(average_effect, binary_df):
    bp = BinaryPerturbator()
    with pytest.raises(ValueError, match="Average effect must be in"):
        bp.perturbate(binary_df, average_effect=average_effect)


def test_binary_raises_non_binary_target(binary_df):
    bp = BinaryPerturbator()
    binary_df["target"] = binary_df["target"] + 0.01
    with pytest.raises(ValueError, match="must be binary"):
        bp.perturbate(binary_df, average_effect=0.05)


def test_stochastic_raises_non_positive_scale(continuous_df):
    _scale = -0.1
    bp = NormalPerturbator(scale=_scale)
    with pytest.raises(ValueError, match=f"scale must be positive, got {_scale}"):
        bp.perturbate(continuous_df, average_effect=0.05)


def test_relative_positive_raises_effect_less_than_minus_100(continuous_df):
    average_effect = -1.1
    rp = RelativePositivePerturbator()
    with pytest.raises(
        ValueError,
        match=f"Simulated effect needs to be greater than -100%, got {average_effect*100:.1f}%",
    ):
        rp.perturbate(continuous_df, average_effect)


def test_relative_positive_target_has_some_negative(continuous_df):
    average_effect = 0.1
    _continuous_df = continuous_df.copy()
    _continuous_df.loc[:, "target"] = [0.5, 0.5, 0.5, -0.1]
    rp = RelativePositivePerturbator()
    msg = "All target values need to be positive or 0, got -0.1"
    with pytest.raises(ValueError, match=msg):
        rp.perturbate(_continuous_df, average_effect)


def test_relative_positive_raises_target_is_all_0(continuous_df):
    average_effect = 0.1
    _continuous_df = continuous_df.copy()
    _continuous_df.loc[_continuous_df["treatment"] == "B", "target"] = 0
    rp = RelativePositivePerturbator()
    msg = (
        "All treatment samples have target = 0, relative effect "
        f"{average_effect} will have no effect"
    )
    with pytest.raises(ValueError, match=msg):
        rp.perturbate(_continuous_df, average_effect)


def test_beta_raises_effect_is_negative(continuous_df):
    average_effect = -0.1
    rp = BetaRelativePositivePerturbator(scale=0.1)
    with pytest.raises(
        ValueError,
        match=f"Simulated effect needs to be greater than 0%, got {average_effect*100:.1f}%",
    ):
        rp.perturbate(continuous_df, average_effect)


def test_beta_raises_effect_equals_0(continuous_df):
    average_effect = 0
    rp = BetaRelativePositivePerturbator(scale=0.1)
    with pytest.raises(
        ValueError,
        match=f"Simulated effect needs to be greater than 0%, got {average_effect*100:.1f}%",
    ):
        rp.perturbate(continuous_df, average_effect)


def test_beta_raises_effect_equals_1(continuous_df):
    average_effect = 1
    rp = BetaRelativePositivePerturbator()
    with pytest.raises(
        ValueError,
        match=f"Simulated effect needs to be less than 100%, got {average_effect*100:.1f}%",
    ):
        rp.perturbate(continuous_df, average_effect)
