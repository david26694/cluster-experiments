from cluster_experiments.pre_experiment_covariates import TargetAggregation

from tests.examples import binary_df


def test_set_target_aggs():
    binary_df["user"] = [1, 1, 1, 1]
    ta = TargetAggregation(agg_col="user")
    ta.fit_pre_experiment_data(binary_df)

    assert len(ta.pre_experiment_agg_df) == 1
    assert ta.pre_experiment_mean == 0.5


def test_smoothing_0():
    binary_df["user"] = binary_df["target"]
    ta = TargetAggregation(agg_col="user", smoothing_factor=0)
    ta.fit_pre_experiment_data(binary_df)
    assert (
        ta.pre_experiment_agg_df["target_mean"]
        == ta.pre_experiment_agg_df["target_smooth_mean"]
    ).all()


def test_smoothing_non_0():
    binary_df["user"] = binary_df["target"]
    ta = TargetAggregation(agg_col="user", smoothing_factor=2)
    ta.fit_pre_experiment_data(binary_df)
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
    ta.fit_pre_experiment_data(binary_df)
    assert (
        ta.add_pre_experiment_data(binary_df).query("user == 0")["target_smooth_mean"]
        == 0.25
    ).all()
