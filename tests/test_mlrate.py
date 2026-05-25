import pandas as pd

from cluster_experiments.cupac import MLHandler, NoOpHandler


def test_noophandler_satisfies_protocol():
    assert isinstance(NoOpHandler(), MLHandler)


def test_noophandler_cupac_outcome_name_is_empty():
    assert NoOpHandler().cupac_outcome_name == ""


def test_noophandler_returns_df_unchanged():
    df = pd.DataFrame({"target": [1, 2, 3], "x": [4, 5, 6]})
    result = NoOpHandler().add_covariates(df)
    pd.testing.assert_frame_equal(result, df)


def test_noophandler_ignores_pre_experiment_df():
    df = pd.DataFrame({"target": [1, 2, 3]})
    pre = pd.DataFrame({"target": [10, 20]})
    result = NoOpHandler().add_covariates(df, pre_experiment_df=pre)
    pd.testing.assert_frame_equal(result, df)
