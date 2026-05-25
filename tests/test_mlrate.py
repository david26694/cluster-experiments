import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from cluster_experiments import NormalPowerAnalysis, OLSAnalysis, PowerAnalysis
from cluster_experiments.cupac import (
    CupacHandler,
    MLHandler,
    MLRateHandler,
    NoOpHandler,
    TargetAggregation,
)
from cluster_experiments.experiment_analysis import ClusteredOLSAnalysis
from cluster_experiments.inference.hypothesis_test import HypothesisTest
from cluster_experiments.inference.metric import SimpleMetric
from cluster_experiments.perturbator import ConstantPerturbator
from cluster_experiments.power_config import PowerConfig
from cluster_experiments.random_splitter import ClusteredSplitter, NonClusteredSplitter


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


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(42)
    n = 100
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "target": 2 * x + rng.normal(scale=0.1, size=n)})


def test_mlrate_satisfies_protocol():
    assert isinstance(
        MLRateHandler(ml_model=LinearRegression(), features=["x"]), MLHandler
    )


def test_mlrate_cupac_outcome_name():
    handler = MLRateHandler(
        ml_model=LinearRegression(), target_col="revenue", features=["x"]
    )
    assert handler.cupac_outcome_name == "estimate_revenue"


def test_mlrate_adds_column(simple_df):
    handler = MLRateHandler(ml_model=LinearRegression(), features=["x"], random_state=0)
    result = handler.add_covariates(simple_df)
    assert "estimate_target" in result.columns
    assert len(result) == len(simple_df)


def test_mlrate_no_data_leakage(simple_df):
    handler = MLRateHandler(
        ml_model=KNeighborsRegressor(n_neighbors=1),
        features=["x"],
        n_folds=5,
        random_state=0,
    )
    result = handler.add_covariates(simple_df)
    assert not np.allclose(result["estimate_target"].values, simple_df["target"].values)


def test_mlrate_missing_features_raises(simple_df):
    handler = MLRateHandler(ml_model=LinearRegression(), features=["nonexistent"])
    with pytest.raises(ValueError, match="nonexistent"):
        handler.add_covariates(simple_df)


def test_mlrate_original_df_not_mutated(simple_df):
    handler = MLRateHandler(ml_model=LinearRegression(), features=["x"], random_state=0)
    original_cols = list(simple_df.columns)
    handler.add_covariates(simple_df)
    assert list(simple_df.columns) == original_cols


@pytest.fixture
def clustered_df():
    rng = np.random.default_rng(0)
    clusters = [f"C{i}" for i in range(20)]
    rows = []
    for c in clusters:
        for _ in range(10):
            x = rng.normal()
            rows.append({"cluster": c, "x": x, "target": 2 * x + rng.normal(scale=0.1)})
    return pd.DataFrame(rows)


def test_mlrate_cluster_no_leakage(clustered_df):
    from sklearn.model_selection import GroupKFold as _GKF

    X = clustered_df[["x"]].values
    y = clustered_df["target"].values
    groups = clustered_df["cluster"].values
    for train_idx, val_idx in _GKF(n_splits=5).split(X, y, groups=groups):
        assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))


def test_mlrate_too_few_clusters_raises():
    df = pd.DataFrame(
        {
            "cluster": ["A", "B", "A", "B", "A", "B"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    handler = MLRateHandler(
        ml_model=LinearRegression(),
        features=["x"],
        cluster_cols=["cluster"],
        n_folds=5,
    )
    with pytest.raises(ValueError, match="n_folds=5 but only 2 unique clusters"):
        handler.add_covariates(df)


def test_mlrate_cluster_adds_column(clustered_df):
    handler = MLRateHandler(
        ml_model=LinearRegression(),
        features=["x"],
        cluster_cols=["cluster"],
        n_folds=5,
    )
    result = handler.add_covariates(clustered_df)
    assert "estimate_target" in result.columns
    assert len(result) == len(clustered_df)


# ---------------------------------------------------------------------------
# Task 4: PowerAnalysis
# ---------------------------------------------------------------------------


@pytest.fixture
def power_df():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "target": 2 * x + rng.normal(scale=0.5, size=n),
            "treatment": rng.choice(["A", "B"], size=n),
        }
    )


def test_power_analysis_default_handler_is_noophandler():
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(),
    )
    assert isinstance(pw.handler, NoOpHandler)


def test_power_analysis_mlrate_builds_handler():
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(covariates=["estimate_target"]),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
    )
    assert isinstance(pw.handler, MLRateHandler)


def test_power_analysis_mlrate_runs(power_df):
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(covariates=["estimate_target"]),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
        n_folds=5,
        features_cupac_model=["x"],
    )
    power = pw.power_analysis(power_df, average_effect=0.5, n_simulations=5)
    assert 0.0 <= power <= 1.0


def test_power_analysis_mlrate_auto_cluster_cols():
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=ClusteredSplitter(cluster_cols=["cluster"]),
        analysis=ClusteredOLSAnalysis(
            covariates=["estimate_target"], cluster_cols=["cluster"]
        ),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
    )
    assert pw.handler.cluster_cols == ["cluster"]


def test_power_analysis_backward_compat_alias():
    pw = PowerAnalysis(
        perturbator=ConstantPerturbator(),
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(),
    )
    assert pw.cupac_handler is pw.handler


# ---------------------------------------------------------------------------
# Task 5: NormalPowerAnalysis
# ---------------------------------------------------------------------------


def test_normal_power_analysis_default_is_noophandler():
    pw = NormalPowerAnalysis(splitter=NonClusteredSplitter(), analysis=OLSAnalysis())
    assert isinstance(pw.handler, NoOpHandler)


def test_normal_power_analysis_mlrate_builds_handler():
    pw = NormalPowerAnalysis(
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(covariates=["estimate_target"]),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
        features_cupac_model=["x"],
    )
    assert isinstance(pw.handler, MLRateHandler)


def test_normal_power_analysis_mlrate_runs(power_df):
    pw = NormalPowerAnalysis(
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(covariates=["estimate_target"]),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
        features_cupac_model=["x"],
        n_simulations=5,
    )
    se = pw._get_average_standard_error(power_df, n_simulations=5)
    assert se > 0


def test_normal_power_analysis_backward_compat_alias():
    pw = NormalPowerAnalysis(splitter=NonClusteredSplitter(), analysis=OLSAnalysis())
    assert pw.cupac_handler is pw.handler


# ---------------------------------------------------------------------------
# Task 6: PowerConfig + cupac_model_mapping + from_config
# ---------------------------------------------------------------------------


def test_power_config_ml_option_and_n_folds():
    config = PowerConfig(
        splitter="non_clustered",
        analysis="ols",
        perturbator="constant",
        cupac_model="linear",
        ml_option="mlrate",
        n_folds=3,
    )
    assert config.ml_option == "mlrate"
    assert config.n_folds == 3


def test_power_analysis_from_dict_mlrate(power_df):
    pw = PowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "perturbator": "constant",
            "cupac_model": "linear",
            "ml_option": "mlrate",
            "n_folds": 3,
            "covariates": ["estimate_target"],
        }
    )
    assert isinstance(pw.handler, MLRateHandler)


def test_power_analysis_from_dict_no_cupac():
    pw = PowerAnalysis.from_dict(
        {
            "splitter": "non_clustered",
            "analysis": "ols",
            "perturbator": "constant",
        }
    )
    assert isinstance(pw.handler, NoOpHandler)


# ---------------------------------------------------------------------------
# Task 7: HypothesisTest + exports + integration
# ---------------------------------------------------------------------------


@pytest.fixture
def metric():
    return SimpleMetric(alias="revenue", name="target")


def test_hypothesis_test_no_cupac_config_is_noophandler(metric):
    test = HypothesisTest(metric=metric, analysis_type="ols")
    assert isinstance(test.handler, NoOpHandler)


def test_hypothesis_test_mlrate(metric, power_df):
    test = HypothesisTest(
        metric=metric,
        analysis_type="ols",
        analysis_config={"covariates": ["estimate_target"]},
        cupac_config={
            "ml_model": LinearRegression(),
            "features": ["x"],
            "target_col": "target",
            "random_state": 0,
        },
        ml_option="mlrate",
    )
    assert isinstance(test.handler, MLRateHandler)
    assert test.handler.cupac_outcome_name == "estimate_target"


def test_hypothesis_test_cupac_unchanged(metric):
    test = HypothesisTest(
        metric=metric,
        analysis_type="ols",
        cupac_config={"cupac_model": TargetAggregation("x"), "target_col": "target"},
    )
    assert isinstance(test.handler, CupacHandler)


def test_hypothesis_test_backward_compat_alias(metric):
    test = HypothesisTest(metric=metric, analysis_type="ols")
    assert test.cupac_handler is test.handler


def test_hypothesis_test_from_config_mlrate(metric):
    config = {
        "metric": {"alias": "revenue", "name": "target"},
        "analysis_type": "ols",
        "analysis_config": {"covariates": ["estimate_target"]},
        "cupac_config": {
            "ml_model": LinearRegression(),
            "features": ["x"],
            "target_col": "target",
        },
        "ml_option": "mlrate",
    }
    test = HypothesisTest.from_config(config)
    assert isinstance(test.handler, MLRateHandler)


def test_mlrate_reduces_variance():
    rng = np.random.default_rng(99)
    n = 2000
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "target": 2 * x + rng.normal(scale=0.2, size=n),
        }
    )
    plain = NormalPowerAnalysis(
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(),
        n_simulations=20,
    )
    mlrate = NormalPowerAnalysis(
        splitter=NonClusteredSplitter(),
        analysis=OLSAnalysis(covariates=["estimate_target"]),
        cupac_model=LinearRegression(),
        ml_option="mlrate",
        features_cupac_model=["x"],
        n_simulations=20,
    )
    se_plain = plain._get_average_standard_error(df, n_simulations=20)
    se_mlrate = mlrate._get_average_standard_error(df, n_simulations=20)
    assert (
        se_mlrate < se_plain
    ), f"Expected MLRATE SE {se_mlrate:.4f} < plain SE {se_plain:.4f}"


def test_mlrate_exported():
    from cluster_experiments import MLHandler, MLRateHandler, NoOpHandler

    assert MLHandler is not None
    assert MLRateHandler is not None
    assert NoOpHandler is not None
