from typing import Tuple

import pandas as pd
from cluster_experiments.cupac import TargetAggregation

from tests.examples import binary_df


def split_x_y(binary_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return binary_df.drop("target", axis=1), binary_df["target"]


def test_set_target_aggs():
    binary_df["user"] = [1, 1, 1, 1]
    ta = TargetAggregation(agg_col="user")
    X, y = split_x_y(binary_df)
    ta.fit(X, y)

    assert len(ta.pre_experiment_agg_df) == 1
    assert ta.pre_experiment_mean == 0.5


def test_smoothing_0():
    binary_df["user"] = binary_df["target"]
    ta = TargetAggregation(agg_col="user", smoothing_factor=0)
    X, y = split_x_y(binary_df)
    ta.fit(X, y)
    assert (
        ta.pre_experiment_agg_df["target_mean"]
        == ta.pre_experiment_agg_df["target_smooth_mean"]
    ).all()


def test_smoothing_non_0():
    binary_df["user"] = binary_df["target"]
    ta = TargetAggregation(agg_col="user", smoothing_factor=2)
    X, y = split_x_y(binary_df)
    ta.fit(X, y)
    assert (
        ta.pre_experiment_agg_df["target_mean"]
        != ta.pre_experiment_agg_df["target_smooth_mean"]
    ).all()
    assert (
        ta.pre_experiment_agg_df["target_smooth_mean"].loc[[0, 1]] == [0.25, 0.75]
    ).all()


def test_add_aggs():
    binary_df["user"] = binary_df["target"]
    ta = TargetAggregation(agg_col="user", smoothing_factor=2)
    X, y = split_x_y(binary_df)
    ta.fit(X, y)
    binary_df["target_smooth_mean"] = ta.predict(binary_df)
    assert (binary_df.query("user == 0")["target_smooth_mean"] == 0.25).all()
